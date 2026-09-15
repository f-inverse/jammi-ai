//! The cross-process collective: rank 0 is the coordinator, in the process
//! that claimed the job; every other rank is a member on the far end of one
//! admitted `RunRank` stream (`jammi.v1.gang`, `crates/jammi-wire/proto/jammi/v1/gang.proto`).
//!
//! # Rank-ordered coordinator-reduce
//!
//! Every member sends its contribution — its [`Descriptor`] and its tensors
//! as Arrow IPC — to the coordinator; the coordinator folds in RANK ORDER on
//! its own device (`Tensor::cat` of the slices for a gather, a left fold of
//! `acc.add(term)` for a sum, `max` for the control word, the root's tensor
//! for a broadcast), the SAME operation sequence [`Local`](super::Local)
//! runs on rank 0's device, so the two arms are byte-identical over the
//! same inputs. The fold is published to every member as one result under
//! the round protocol below, and every rank leaves the round holding the
//! same bytes. Elements travel exactly: f32 as an Arrow `Float32` column,
//! f16 as `Float16`, bf16 as its `UInt16` bit pattern (Arrow has no bf16);
//! every other candle dtype is refused at the seam before a descriptor is
//! built. A payload larger than the listener's `[server.limits]
//! max_message_bytes` is split into `RoundChunk`s of at most that size and
//! reassembled on the far side against the byte bound the descriptor
//! implies.
//!
//! # The two-phase round
//!
//! [`super`]'s module doc names the state the in-process arm can afford and
//! the wire cannot: a round that published and then faulted before every
//! rank took its result. This arm decides it with a commit point. A round
//! `k` on a member is: send the contribution; wait for the result; decode
//! it and HOLD it unapplied; send `RoundAck`; wait for `RoundCommit`; apply.
//! On the coordinator: collect every member's contribution; refuse the round
//! on EVERY rank (a `RoundFault` to each member, a typed error here) unless
//! every descriptor equals rank 0's; fold; publish; wait for every member's
//! ACK; send `RoundCommit` to every member; apply. So a fault before the
//! coordinator has observed the last ACK — a disconnect, a fault raised by
//! any rank, the gang deadline — leaves NO rank applied for round `k`, and
//! every rank's error names `k`. A fault DURING the commit fan-out is fatal
//! on every rank too: the coordinator applies nothing, sends a `RoundFault`
//! to every member and refuses every later round; a member whose stream
//! ended between its ACK and the commit applied nothing; a member the
//! commit did reach applied `k` and returned `Ok` — that cannot be
//! retracted — but its next contribution is answered by the coordinator's
//! fault, so no rank ever CONTINUES past a round every rank did not apply.
//! That one residual state (one member applied, the gang faulted before
//! its next round) is stated here rather than inherited from the
//! shared-memory arm.
//!
//! The round index is a field of the agreed [`Descriptor`] on the wire: a
//! contribution from a stale round is a typed disagreement naming both
//! descriptors, never folded.
//!
//! Every wait on every rank — for a contribution, a result, a chunk, an
//! ACK, a commit — expires at the gang deadline ([`Peer::with_timeout`],
//! the same bound as the in-process arm's rendezvous) with a typed error
//! naming the round and what it waited for. A fault is permanent: every
//! later verb on the faulted rank refuses, quoting the fault, exactly as
//! the in-process arm does. A rank that refuses its own arguments (a counts
//! vector that is not `world`-long, a root outside the gang) faults the
//! round for its peers too — a `RoundFault` naming the round goes out
//! before the typed error comes back — so no peer waits out the deadline
//! for a contribution that was never going to come.
//!
//! # Links
//!
//! [`Peer`] never opens a stream itself. A [`MemberLink`] is the member's
//! end of one admitted `RunRank` stream (built by the handler that admitted
//! it, over the inbound `Streaming<RankControl>` and the outbound event
//! sender); a [`CoordinatorLink`] is the coordinator's end (built over a
//! `GangServiceClient` once the member answered `Admitted`). Both are also
//! constructible over plain channels, which is how the hermetic oracles
//! drive a whole gang in one process through the identical round code —
//! the only code a stream adds is the pump that moves frames between the
//! tonic stream and the channel.
//!
//! # Blocking
//!
//! Every verb blocks the calling thread on stream I/O through the
//! [`tokio::runtime::Handle`] its links captured — which is why every verb
//! takes a [`BlockingCall`]: a runtime worker thread cannot mint one, so
//! the `block_on` that would panic there is unreachable by construction.
//!
//! # The rank's read path
//!
//! Before its first collective a rank reads its partition of the training
//! set from the shared object store and verifies each row group it reads
//! against the attestation's leaf inventory ([`verify_partition_leaves`]) —
//! one leaf at a time through a ranged read, so memory is bounded by the
//! largest row group, never the artifact. A failure there is
//! [`RankReadFault::StoreUnavailable`]: MEMBER-scoped, the host's own store
//! did not hand back the attested bytes, mapped to the wire's
//! `ABORT_REASON_STORE_UNAVAILABLE` — never an assembly refutation, never
//! counted against the attempt budget.

use std::fmt;
use std::ops::Range;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use arrow::array::{Array, ArrayRef, Float16Array, Float32Array, UInt16Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use candle_core::{DType, Device, Tensor};
use jammi_db::error::{JammiError, Result};
use jammi_db::storage::JammiObjectStore;
use jammi_db::store::manifest::{ArtifactDigest, LeafDigest, LeafKey};
use jammi_wire::proto::gang::gang_service_client::GangServiceClient;
use jammi_wire::proto::gang::{
    rank_control, rank_event, AbortReason, Assign, Counts, RankControl, RankEvent, RoundAck,
    RoundChunk, RoundCommit, RoundDescriptor, RoundFault, RoundPayload, RoundVerb,
};
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::transport::Channel;

use super::local::DEFAULT_RENDEZVOUS_TIMEOUT;
use super::{
    checked_gather_counts, checked_root, BlockingCall, Collective, Descriptor, TensorSignature,
    Verb,
};

/// Frames buffered per direction of one link before the sender waits.
const LINK_CAPACITY: usize = 64;

/// Bytes a `RoundChunk` frame spends outside its `ipc` payload (the three
/// field tags, two varints, the length prefix), rounded up.
const CHUNK_FRAME_OVERHEAD: usize = 64;

/// The smallest `max_message_bytes` a gang message cap may be: below this a
/// chunk would carry no payload at all.
const MIN_MESSAGE_CAP: usize = 1024;

/// Slack the reassembly bound allows per tensor beyond the elements' own
/// bytes: the IPC schema, message headers and 64-byte buffer alignment.
const IPC_SLACK_PER_TENSOR: usize = 4096;

// ── Links ───────────────────────────────────────────────────────────────────

/// Why a link's inbound side ended with an error rather than a clean close:
/// the transport's own words (a `tonic::Status` from the stream).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkFault(pub String);

/// One end of one stream: frames in (a clean close is `None`, a transport
/// fault is `Err`), frames out, and the runtime the stream lives on.
struct Link<In, Out> {
    inbound: mpsc::Receiver<std::result::Result<In, LinkFault>>,
    outbound: mpsc::Sender<Out>,
    handle: Handle,
}

impl<In, Out> fmt::Debug for Link<In, Out> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Link").finish_non_exhaustive()
    }
}

/// The runtime the link's frames move on — captured where the link is
/// built (always inside a runtime context: a handler, a client task, a
/// test's `block_on`), so the verbs can block on it from any non-worker
/// thread.
fn current_handle(what: &str) -> Result<Handle> {
    Handle::try_current().map_err(|_| {
        JammiError::FineTune(format!(
            "{what}: a Peer link must be built inside a tokio runtime context — the stream it \
             wraps lives on that runtime"
        ))
    })
}

/// Move a tonic stream's items onto a channel: a transport error is
/// forwarded as a [`LinkFault`] and ends the pump; a clean end closes the
/// channel.
fn pump<In: Send + 'static>(
    handle: &Handle,
    mut stream: tonic::Streaming<In>,
) -> mpsc::Receiver<std::result::Result<In, LinkFault>> {
    let (tx, rx) = mpsc::channel(LINK_CAPACITY);
    handle.spawn(async move {
        while let Some(item) = stream.next().await {
            let item = item.map_err(|status| LinkFault(status.to_string()));
            let ended = item.is_err();
            if tx.send(item).await.is_err() || ended {
                break;
            }
        }
    });
    rx
}

/// A member's end of one admitted `RunRank` stream: `RankControl` frames in
/// from the coordinator, `RankEvent` frames out to it.
#[derive(Debug)]
pub struct MemberLink(Link<RankControl, RankEvent>);

impl MemberLink {
    /// A link over plain channels — what the hermetic oracles drive a whole
    /// gang through in one process; the round code is identical.
    pub fn from_channels(
        inbound: mpsc::Receiver<std::result::Result<RankControl, LinkFault>>,
        outbound: mpsc::Sender<RankEvent>,
    ) -> Result<Self> {
        Ok(Self(Link {
            inbound,
            outbound,
            handle: current_handle("MemberLink::from_channels")?,
        }))
    }
}

/// The coordinator's end of one member's admitted `RunRank` stream:
/// `RankEvent` frames in from the member, `RankControl` frames out to it,
/// tagged with the member's rank.
#[derive(Debug)]
pub struct CoordinatorLink {
    rank: u32,
    link: Link<RankEvent, RankControl>,
}

impl CoordinatorLink {
    /// A link over plain channels for the member at `rank` — see
    /// [`MemberLink::from_channels`].
    pub fn from_channels(
        rank: u32,
        inbound: mpsc::Receiver<std::result::Result<RankEvent, LinkFault>>,
        outbound: mpsc::Sender<RankControl>,
    ) -> Result<Self> {
        Ok(Self {
            rank,
            link: Link {
                inbound,
                outbound,
                handle: current_handle("CoordinatorLink::from_channels")?,
            },
        })
    }

    /// Open `RunRank` over `channel` for the member `assign` names, send the
    /// `Assign`, and require the member's first frame to be `Admitted`: any
    /// other first frame — an `Aborted{reason}`, a status, a close — is a
    /// typed error naming it, and no link exists. The client's OWN inbound
    /// decode cap is `max_message_bytes` — the same `[server.limits]` value
    /// every listener applies, so a member's frame is bounded identically
    /// in both directions (tonic's default would otherwise cap this side
    /// at 4 MiB regardless of the deployment's setting).
    pub async fn over_client(
        channel: Channel,
        assign: Assign,
        max_message_bytes: usize,
    ) -> Result<Self> {
        let rank = assign.rank;
        let handle = current_handle("CoordinatorLink::over_client")?;
        let mut client =
            GangServiceClient::new(channel).max_decoding_message_size(max_message_bytes);
        let (out_tx, out_rx) = mpsc::channel::<RankControl>(LINK_CAPACITY);
        out_tx
            .send(RankControl {
                control: Some(rank_control::Control::Assign(assign)),
            })
            .await
            .map_err(|_| JammiError::FineTune("RunRank: the outbound channel closed".into()))?;
        let mut inbound = client
            .run_rank(ReceiverStream::new(out_rx))
            .await
            .map_err(|status| {
                JammiError::FineTune(format!("RunRank to rank {rank} was refused: {status}"))
            })?
            .into_inner();
        match inbound.next().await {
            Some(Ok(RankEvent {
                event: Some(rank_event::Event::Admitted(_)),
            })) => {}
            Some(Ok(RankEvent {
                event: Some(rank_event::Event::Aborted(aborted)),
            })) => {
                return Err(JammiError::FineTune(format!(
                    "RunRank to rank {rank} was aborted before admission: {}",
                    abort_reason(aborted.reason)
                )));
            }
            Some(Ok(other)) => {
                return Err(JammiError::FineTune(format!(
                    "RunRank to rank {rank}: the first frame must be Admitted, got {other:?}"
                )));
            }
            Some(Err(status)) => {
                return Err(JammiError::FineTune(format!(
                    "RunRank to rank {rank} failed before admission: {status}"
                )));
            }
            None => {
                return Err(JammiError::FineTune(format!(
                    "RunRank to rank {rank} closed before admission"
                )));
            }
        }
        let inbound = pump(&handle, inbound);
        Ok(Self {
            rank,
            link: Link {
                inbound,
                outbound: out_tx,
                handle,
            },
        })
    }

    /// The member this link reaches.
    pub fn rank(&self) -> u32 {
        self.rank
    }
}

/// The wire's `AbortReason` for a raw enum value, or the raw value when it
/// is outside the frozen set.
fn abort_reason(raw: i32) -> String {
    match AbortReason::try_from(raw) {
        Ok(reason) => format!("{reason:?}"),
        Err(_) => format!("unknown abort reason {raw}"),
    }
}

// ── Frames ──────────────────────────────────────────────────────────────────

/// The round protocol's view of one inbound frame, whichever direction it
/// came from. Every oneof arm of both wire frames maps here exhaustively —
/// a frame that belongs to the admission handshake rather than to a round
/// is `Session`, described, and a round refuses it as a protocol violation
/// rather than skipping it.
#[derive(Debug)]
enum Frame {
    Payload(RoundPayload),
    Chunk(RoundChunk),
    Ack(RoundAck),
    Commit(RoundCommit),
    Fault(RoundFault),
    Session(String),
}

/// A wire frame a round can read.
trait Inbound: Send + 'static {
    fn into_frame(self) -> Frame;
}

impl Inbound for RankControl {
    fn into_frame(self) -> Frame {
        match self.control {
            Some(rank_control::Control::RoundResult(payload)) => Frame::Payload(payload),
            Some(rank_control::Control::RoundChunk(chunk)) => Frame::Chunk(chunk),
            Some(rank_control::Control::RoundCommit(commit)) => Frame::Commit(commit),
            Some(rank_control::Control::RoundFault(fault)) => Frame::Fault(fault),
            Some(rank_control::Control::Assign(_)) => {
                Frame::Session("a second Assign on an admitted stream".into())
            }
            Some(rank_control::Control::Cancel(_)) => {
                Frame::Session("Cancel — the coordinator ended this session".into())
            }
            None => Frame::Session("an empty RankControl frame".into()),
        }
    }
}

impl Inbound for RankEvent {
    fn into_frame(self) -> Frame {
        match self.event {
            Some(rank_event::Event::RoundContribution(payload)) => Frame::Payload(payload),
            Some(rank_event::Event::RoundChunk(chunk)) => Frame::Chunk(chunk),
            Some(rank_event::Event::RoundAck(ack)) => Frame::Ack(ack),
            Some(rank_event::Event::RoundFault(fault)) => Frame::Fault(fault),
            Some(rank_event::Event::Admitted(_)) => {
                Frame::Session("a second Admitted on an admitted stream".into())
            }
            Some(rank_event::Event::Aborted(aborted)) => Frame::Session(format!(
                "Aborted({}) — the member ended its session",
                abort_reason(aborted.reason)
            )),
            Some(rank_event::Event::Outcome(_)) => {
                Frame::Session("Outcome — the member's rank body ended".into())
            }
            None => Frame::Session("an empty RankEvent frame".into()),
        }
    }
}

/// A wire frame a round can write — the three both directions share.
trait Outbound: Send + 'static {
    fn payload(payload: RoundPayload) -> Self;
    fn chunk(chunk: RoundChunk) -> Self;
    fn fault(fault: RoundFault) -> Self;
}

impl Outbound for RankEvent {
    fn payload(payload: RoundPayload) -> Self {
        Self {
            event: Some(rank_event::Event::RoundContribution(payload)),
        }
    }
    fn chunk(chunk: RoundChunk) -> Self {
        Self {
            event: Some(rank_event::Event::RoundChunk(chunk)),
        }
    }
    fn fault(fault: RoundFault) -> Self {
        Self {
            event: Some(rank_event::Event::RoundFault(fault)),
        }
    }
}

impl Outbound for RankControl {
    fn payload(payload: RoundPayload) -> Self {
        Self {
            control: Some(rank_control::Control::RoundResult(payload)),
        }
    }
    fn chunk(chunk: RoundChunk) -> Self {
        Self {
            control: Some(rank_control::Control::RoundChunk(chunk)),
        }
    }
    fn fault(fault: RoundFault) -> Self {
        Self {
            control: Some(rank_control::Control::RoundFault(fault)),
        }
    }
}

/// How a bounded wait on a link ended without a frame.
enum WaitEnd {
    /// The far side closed the stream cleanly.
    Disconnected,
    /// The transport reported an error.
    Transport(String),
    /// The gang deadline passed.
    Timeout,
}

impl<In: Inbound, Out: Outbound> Link<In, Out> {
    /// One bounded receive: the next frame, or how the wait ended.
    fn recv(&mut self, deadline: Instant) -> std::result::Result<Frame, WaitEnd> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(WaitEnd::Timeout);
        }
        let inbound = &mut self.inbound;
        // The timer is built INSIDE the runtime context `block_on` enters —
        // `tokio::time::timeout` arms its `Sleep` eagerly, and a blocking
        // thread has no ambient timer of its own.
        match self
            .handle
            .block_on(async { tokio::time::timeout(remaining, inbound.recv()).await })
        {
            Ok(Some(Ok(frame))) => Ok(frame.into_frame()),
            Ok(Some(Err(LinkFault(reason)))) => Err(WaitEnd::Transport(reason)),
            Ok(None) => Err(WaitEnd::Disconnected),
            Err(_elapsed) => Err(WaitEnd::Timeout),
        }
    }

    /// One send; `false` when the far side is gone.
    fn send(&self, frame: Out) -> bool {
        self.handle.block_on(self.outbound.send(frame)).is_ok()
    }

    /// Send a payload and its chunks; `false` at the first send the far
    /// side did not take.
    fn send_payload(&self, round: u64, descriptor: RoundDescriptor, body: &Body) -> bool {
        let chunks = split_chunks(&body.ipc, body.chunk_bytes);
        let payload = RoundPayload {
            round,
            descriptor: Some(descriptor),
            chunk_count: chunks.len() as u32,
            flags: body.flags,
        };
        if !self.send(Out::payload(payload)) {
            return false;
        }
        for (index, ipc) in chunks.into_iter().enumerate() {
            let chunk = RoundChunk {
                round,
                index: index as u32,
                ipc,
            };
            if !self.send(Out::chunk(chunk)) {
                return false;
            }
        }
        true
    }

    fn send_fault(&self, round: u64, detail: &str) {
        let _ = self.send(Out::fault(RoundFault {
            round,
            detail: detail.to_string(),
        }));
    }
}

// ── Wire conversions (K2 on every numeric edge) ─────────────────────────────

fn verb_to_wire(verb: Verb) -> RoundVerb {
    match verb {
        Verb::AllGather => RoundVerb::AllGather,
        Verb::AllReduceSum => RoundVerb::AllReduceSum,
        Verb::AllReduceMaxFlags => RoundVerb::AllReduceMaxFlags,
        Verb::Broadcast => RoundVerb::Broadcast,
        Verb::Barrier => RoundVerb::Barrier,
    }
}

/// The verb a raw wire value names, or the refusal: `UNSPECIFIED` and every
/// value outside the frozen set are unknown, never a default.
fn verb_from_wire(raw: i32) -> std::result::Result<Verb, String> {
    match RoundVerb::try_from(raw) {
        Ok(RoundVerb::AllGather) => Ok(Verb::AllGather),
        Ok(RoundVerb::AllReduceSum) => Ok(Verb::AllReduceSum),
        Ok(RoundVerb::AllReduceMaxFlags) => Ok(Verb::AllReduceMaxFlags),
        Ok(RoundVerb::Broadcast) => Ok(Verb::Broadcast),
        Ok(RoundVerb::Barrier) => Ok(Verb::Barrier),
        Ok(RoundVerb::Unspecified) | Err(_) => Err(format!("unknown wire verb {raw}")),
    }
}

/// The wire dtype of a tensor this arm can carry. Every candle dtype is
/// named: the three this arm carries map, every other one is refused
/// before a descriptor exists. `candle_core::DType` is `#[non_exhaustive]`,
/// so the wildcard arm is mandatory — it covers a dtype a later candle adds,
/// and refuses it the same way, so no dtype can reach the codec unnamed.
fn dtype_to_wire(
    dtype: DType,
) -> std::result::Result<jammi_wire::proto::gang::ElementType, String> {
    use jammi_wire::proto::gang::ElementType as Wire;
    match dtype {
        DType::F32 => Ok(Wire::F32),
        DType::F16 => Ok(Wire::F16),
        DType::BF16 => Ok(Wire::Bf16),
        DType::U8
        | DType::U32
        | DType::I16
        | DType::I32
        | DType::I64
        | DType::F64
        | DType::F8E4M3
        | _ => Err(format!(
            "the Peer collective carries f32, f16 and bf16 tensors only; {dtype:?} is refused \
             at the seam"
        )),
    }
}

fn dtype_from_wire(raw: i32) -> std::result::Result<DType, String> {
    use jammi_wire::proto::gang::ElementType as Wire;
    match Wire::try_from(raw) {
        Ok(Wire::F32) => Ok(DType::F32),
        Ok(Wire::F16) => Ok(DType::F16),
        Ok(Wire::Bf16) => Ok(DType::BF16),
        Ok(Wire::Unspecified) | Err(_) => Err(format!("unknown wire dtype {raw}")),
    }
}

fn descriptor_to_wire(descriptor: &Descriptor) -> std::result::Result<RoundDescriptor, String> {
    let world = u32::try_from(descriptor.world)
        .map_err(|_| format!("world {} does not fit the wire", descriptor.world))?;
    let mut tensors = Vec::with_capacity(descriptor.tensors.len());
    for signature in &descriptor.tensors {
        tensors.push(jammi_wire::proto::gang::TensorSignature {
            dims: signature.dims.iter().map(|&d| d as u64).collect(),
            dtype: dtype_to_wire(signature.dtype)? as i32,
        });
    }
    Ok(RoundDescriptor {
        round: descriptor.round,
        verb: verb_to_wire(descriptor.verb) as i32,
        world,
        root: descriptor.root,
        counts: descriptor.counts.as_ref().map(|counts| Counts {
            values: counts.iter().map(|&c| c as u64).collect(),
        }),
        tensors,
        agreement: descriptor.agreement.clone(),
    })
}

/// Decode a peer's descriptor, range-checking every numeric field: an
/// unknown verb or dtype, a root outside the world, a count or dim that
/// does not fit `usize`, or a shape whose element count overflows is a
/// refusal naming the offending field, and the caller names both sides.
fn descriptor_from_wire(wire: &RoundDescriptor) -> std::result::Result<Descriptor, String> {
    let verb = verb_from_wire(wire.verb)?;
    let world =
        usize::try_from(wire.world).map_err(|_| format!("world {} overflows", wire.world))?;
    if let Some(root) = wire.root {
        if root >= wire.world {
            return Err(format!(
                "root {root} is not a rank of a gang of {}",
                wire.world
            ));
        }
    }
    let counts = match &wire.counts {
        Some(counts) => {
            let mut values = Vec::with_capacity(counts.values.len());
            for &value in &counts.values {
                values.push(
                    usize::try_from(value).map_err(|_| format!("count {value} overflows usize"))?,
                );
            }
            Some(values)
        }
        None => None,
    };
    let mut tensors = Vec::with_capacity(wire.tensors.len());
    for signature in &wire.tensors {
        let mut dims = Vec::with_capacity(signature.dims.len());
        for &dim in &signature.dims {
            dims.push(usize::try_from(dim).map_err(|_| format!("dim {dim} overflows usize"))?);
        }
        element_count(&dims)?;
        tensors.push(TensorSignature {
            dims,
            dtype: dtype_from_wire(signature.dtype)?,
        });
    }
    Ok(Descriptor {
        round: wire.round,
        verb,
        world,
        root: wire.root,
        counts,
        tensors,
        agreement: wire.agreement.clone(),
    })
}

/// The element count of a shape, overflow-checked.
fn element_count(dims: &[usize]) -> std::result::Result<usize, String> {
    dims.iter().try_fold(1usize, |acc, &d| {
        acc.checked_mul(d)
            .ok_or_else(|| format!("shape {dims:?} has an element count that overflows usize"))
    })
}

// ── Payload codec (Arrow IPC, exact) ────────────────────────────────────────

/// The bytes one rank sends for one round: the tensors as a sequence of
/// length-prefixed Arrow IPC streams (one stream, one single-column batch
/// per tensor — tensors of one round differ in length and dtype, and an
/// IPC stream carries one schema), plus the control word for the one verb
/// that carries no tensor, and the chunk size the link splits it at.
struct Body {
    ipc: Vec<u8>,
    flags: u32,
    chunk_bytes: usize,
}

/// The shape and dtype one wire tensor decodes to.
type Shape = (Vec<usize>, DType);

/// The exact bytes of `t`'s elements as an Arrow column of its dtype.
fn tensor_to_column(t: &Tensor) -> Result<ArrayRef> {
    let host = t
        .to_device(&Device::Cpu)
        .and_then(|t| t.flatten_all())
        .map_err(|e| JammiError::FineTune(format!("Peer: to host: {e}")))?;
    let column: ArrayRef = match host.dtype() {
        DType::F32 => std::sync::Arc::new(Float32Array::from(
            host.to_vec1::<f32>()
                .map_err(|e| JammiError::FineTune(format!("Peer: read f32: {e}")))?,
        )),
        DType::F16 => std::sync::Arc::new(Float16Array::from(
            host.to_vec1::<half::f16>()
                .map_err(|e| JammiError::FineTune(format!("Peer: read f16: {e}")))?,
        )),
        DType::BF16 => std::sync::Arc::new(UInt16Array::from(
            host.to_vec1::<half::bf16>()
                .map_err(|e| JammiError::FineTune(format!("Peer: read bf16: {e}")))?
                .into_iter()
                .map(half::bf16::to_bits)
                .collect::<Vec<u16>>(),
        )),
        other => {
            // Unreachable through the verbs: `dtype_to_wire` refused this
            // dtype while building the descriptor, before any encoding.
            return Err(JammiError::FineTune(format!(
                "Peer: cannot encode a {other:?} tensor"
            )));
        }
    };
    Ok(column)
}

/// Encode `tensors` as a body's IPC bytes.
fn encode_tensors(tensors: &[&Tensor]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for t in tensors {
        let column = tensor_to_column(t)?;
        let schema = Schema::new(vec![Field::new("t", column.data_type().clone(), false)]);
        let batch = RecordBatch::try_new(std::sync::Arc::new(schema.clone()), vec![column])
            .map_err(|e| JammiError::FineTune(format!("Peer: build batch: {e}")))?;
        let mut stream = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut stream, &schema)
                .map_err(|e| JammiError::FineTune(format!("Peer: ipc writer: {e}")))?;
            writer
                .write(&batch)
                .and_then(|()| writer.finish())
                .map_err(|e| JammiError::FineTune(format!("Peer: ipc write: {e}")))?;
        }
        let len = u32::try_from(stream.len()).map_err(|_| {
            JammiError::FineTune(format!(
                "Peer: one tensor's IPC stream is {} bytes, past the 4 GiB frame length",
                stream.len()
            ))
        })?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&stream);
    }
    Ok(out)
}

/// Decode a body's IPC bytes into one host tensor per `expected` shape,
/// refusing a stream whose column dtype or element count is not the one
/// the (already agreed) descriptor promised, or bytes left over.
fn decode_tensors(
    mut bytes: &[u8],
    expected: &[Shape],
) -> std::result::Result<Vec<Tensor>, String> {
    let mut tensors = Vec::with_capacity(expected.len());
    for (index, (dims, dtype)) in expected.iter().enumerate() {
        let Some(prefix) = bytes.get(..4) else {
            return Err(format!(
                "tensor {index}: the payload ended before its length prefix"
            ));
        };
        let len = u32::from_le_bytes([prefix[0], prefix[1], prefix[2], prefix[3]]) as usize;
        bytes = &bytes[4..];
        let Some(stream) = bytes.get(..len) else {
            return Err(format!(
                "tensor {index}: the payload holds {} bytes where its prefix says {len}",
                bytes.len()
            ));
        };
        bytes = &bytes[len..];
        let mut reader = StreamReader::try_new(std::io::Cursor::new(stream), None)
            .map_err(|e| format!("tensor {index}: ipc reader: {e}"))?;
        let batch = match reader.next() {
            Some(Ok(batch)) => batch,
            Some(Err(e)) => return Err(format!("tensor {index}: ipc read: {e}")),
            None => return Err(format!("tensor {index}: the IPC stream holds no batch")),
        };
        if reader.next().is_some() {
            return Err(format!(
                "tensor {index}: the IPC stream holds more than one batch"
            ));
        }
        if batch.num_columns() != 1 {
            return Err(format!(
                "tensor {index}: the batch holds {} columns, not one",
                batch.num_columns()
            ));
        }
        let column = batch.column(0);
        let count = element_count(dims)?;
        if column.len() != count {
            return Err(format!(
                "tensor {index}: {} elements on the wire where the agreed shape {dims:?} holds \
                 {count}",
                column.len()
            ));
        }
        let tensor = match (dtype, column.data_type()) {
            (DType::F32, DataType::Float32) => {
                let values = column
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| format!("tensor {index}: not a Float32 column"))?
                    .values()
                    .to_vec();
                Tensor::from_vec(values, dims.as_slice(), &Device::Cpu)
            }
            (DType::F16, DataType::Float16) => {
                let values = column
                    .as_any()
                    .downcast_ref::<Float16Array>()
                    .ok_or_else(|| format!("tensor {index}: not a Float16 column"))?
                    .values()
                    .to_vec();
                Tensor::from_vec(values, dims.as_slice(), &Device::Cpu)
            }
            (DType::BF16, DataType::UInt16) => {
                let values: Vec<half::bf16> = column
                    .as_any()
                    .downcast_ref::<UInt16Array>()
                    .ok_or_else(|| format!("tensor {index}: not a UInt16 column"))?
                    .values()
                    .iter()
                    .map(|&bits| half::bf16::from_bits(bits))
                    .collect();
                Tensor::from_vec(values, dims.as_slice(), &Device::Cpu)
            }
            (dtype, wire) => {
                return Err(format!(
                    "tensor {index}: the agreed dtype is {dtype:?} but the wire column is {wire:?}"
                ));
            }
        }
        .map_err(|e| format!("tensor {index}: build: {e}"))?;
        tensors.push(tensor);
    }
    if !bytes.is_empty() {
        return Err(format!(
            "{} bytes left over after the last agreed tensor",
            bytes.len()
        ));
    }
    Ok(tensors)
}

/// The most bytes a body for `expected` may reassemble to: the elements'
/// own bytes plus per-tensor IPC slack. A peer announcing more bytes than
/// this bound admits is refused before they are buffered.
fn reassembly_bound(expected: &[Shape]) -> std::result::Result<usize, String> {
    let mut bound = IPC_SLACK_PER_TENSOR;
    for (dims, dtype) in expected {
        let bytes = element_count(dims)?
            .checked_mul(dtype.size_in_bytes())
            .and_then(|b| b.checked_add(IPC_SLACK_PER_TENSOR))
            .ok_or_else(|| format!("shape {dims:?} has a byte size that overflows usize"))?;
        bound = bound
            .checked_add(bytes)
            .ok_or_else(|| "the payload's byte bound overflows usize".to_string())?;
    }
    Ok(bound)
}

fn split_chunks(ipc: &[u8], chunk_bytes: usize) -> Vec<Vec<u8>> {
    ipc.chunks(chunk_bytes).map(<[u8]>::to_vec).collect()
}

// ── Contributions ───────────────────────────────────────────────────────────

/// What one rank contributes to one round, on the host.
enum Contribution {
    Gather(Tensor),
    ReduceSum(Vec<Tensor>),
    MaxFlags(u32),
    Broadcast(Option<Tensor>),
    Barrier,
}

impl Contribution {
    fn tensors(&self) -> Vec<&Tensor> {
        match self {
            Self::Gather(t) => vec![t],
            Self::ReduceSum(ts) => ts.iter().collect(),
            Self::MaxFlags(_) | Self::Barrier => Vec::new(),
            Self::Broadcast(t) => t.iter().collect(),
        }
    }

    fn flags(&self) -> u32 {
        match self {
            Self::MaxFlags(flags) => *flags,
            _ => 0,
        }
    }
}

/// The shapes a contribution from `rank` carries under `descriptor` — the
/// descriptor is agreed before any body is decoded, so this is what the
/// body MUST decode to.
fn contribution_shapes(descriptor: &Descriptor, rank: u32) -> Vec<Shape> {
    match descriptor.verb {
        Verb::AllGather => {
            let rows = descriptor
                .counts
                .as_ref()
                .and_then(|counts| counts.get(rank as usize).copied())
                .unwrap_or(0);
            gather_shapes(descriptor, rows)
        }
        Verb::AllReduceSum => full_shapes(descriptor),
        Verb::Broadcast => {
            if descriptor.root == Some(rank) {
                full_shapes(descriptor)
            } else {
                Vec::new()
            }
        }
        Verb::AllReduceMaxFlags | Verb::Barrier => Vec::new(),
    }
}

/// The shapes the RESULT of a round under `descriptor` carries.
fn result_shapes(descriptor: &Descriptor) -> Vec<Shape> {
    match descriptor.verb {
        Verb::AllGather => {
            let total: usize = descriptor
                .counts
                .as_ref()
                .map(|counts| counts.iter().sum())
                .unwrap_or(0);
            gather_shapes(descriptor, total)
        }
        Verb::AllReduceSum | Verb::Broadcast => full_shapes(descriptor),
        Verb::AllReduceMaxFlags | Verb::Barrier => Vec::new(),
    }
}

fn gather_shapes(descriptor: &Descriptor, rows: usize) -> Vec<Shape> {
    descriptor
        .tensors
        .iter()
        .map(|s| {
            let mut dims = Vec::with_capacity(s.dims.len() + 1);
            dims.push(rows);
            dims.extend_from_slice(&s.dims);
            (dims, s.dtype)
        })
        .collect()
}

fn full_shapes(descriptor: &Descriptor) -> Vec<Shape> {
    descriptor
        .tensors
        .iter()
        .map(|s| (s.dims.clone(), s.dtype))
        .collect()
}

/// A decoded round result, held UNAPPLIED on the host until the commit.
struct Pending {
    tensors: Vec<Tensor>,
    flags: u32,
}

// ── The Peer ────────────────────────────────────────────────────────────────

/// Which end of the stream(s) this rank is.
enum Endpoint {
    /// Rank 0: one link per member, in rank order 1..world.
    Coordinator(Vec<CoordinatorLink>),
    /// A member: its one link to the coordinator.
    Member(MemberLink),
}

struct Inner {
    endpoint: Endpoint,
    /// The next round this rank enters; every rank counts from 0.
    next_round: u64,
    /// The first failure this rank saw, and its permanent state from then
    /// on: every later verb refuses quoting it.
    fault: Option<String>,
}

/// The cross-process collective — one value per rank, see the module doc.
///
/// Verbs serialize on an internal lock (one rank runs one collective at a
/// time by construction; the lock is what lets the trait's `&self` verbs
/// drive the links' `&mut` receivers).
pub struct Peer {
    rank: u32,
    world: u32,
    device: Device,
    timeout: Duration,
    chunk_bytes: usize,
    agreement: Option<String>,
    inner: Mutex<Inner>,
}

impl fmt::Debug for Peer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Peer")
            .field("rank", &self.rank)
            .field("world", &self.world)
            .field("timeout", &self.timeout)
            .field("chunk_bytes", &self.chunk_bytes)
            .field("agreement", &self.agreement)
            .finish_non_exhaustive()
    }
}

impl Peer {
    /// Rank 0 over one admitted link per member. `members` must hold ranks
    /// `1..=members.len()` exactly once each (in any order; they are sorted
    /// here), so `world = members.len() + 1`; a gang of one never crosses
    /// the wire and is refused. `max_message_bytes` is the listener's
    /// `[server.limits] max_message_bytes` — every chunk fits under it.
    pub fn coordinator(
        members: Vec<CoordinatorLink>,
        device: Device,
        max_message_bytes: usize,
    ) -> Result<Self> {
        if members.is_empty() {
            return Err(JammiError::FineTune(
                "a Peer coordinator needs at least one member: a gang of one never crosses the \
                 wire (it holds a Noop)"
                    .into(),
            ));
        }
        let mut members = members;
        members.sort_by_key(CoordinatorLink::rank);
        let world = u32::try_from(members.len() + 1)
            .map_err(|_| JammiError::FineTune("world overflows u32".into()))?;
        for (index, link) in members.iter().enumerate() {
            let expected = index as u32 + 1;
            if link.rank != expected {
                return Err(JammiError::FineTune(format!(
                    "a Peer coordinator's members must be ranks 1..{world} exactly once each; \
                     found rank {} where rank {expected} was expected",
                    link.rank
                )));
            }
        }
        Ok(Self {
            rank: 0,
            world,
            device,
            timeout: DEFAULT_RENDEZVOUS_TIMEOUT,
            chunk_bytes: chunk_bytes(max_message_bytes)?,
            agreement: None,
            inner: Mutex::new(Inner {
                endpoint: Endpoint::Coordinator(members),
                next_round: 0,
                fault: None,
            }),
        })
    }

    /// The member at `rank` of a gang of `world`, over its admitted link.
    /// `rank` must be `1..world` — rank 0 is the coordinator, and a rank
    /// outside the gang is refused before a `Peer` exists.
    pub fn member(
        rank: u32,
        world: u32,
        link: MemberLink,
        device: Device,
        max_message_bytes: usize,
    ) -> Result<Self> {
        if world < 2 {
            return Err(JammiError::FineTune(format!(
                "a Peer member needs a gang of at least two: world {world} never crosses the wire"
            )));
        }
        if rank == 0 || rank >= world {
            return Err(JammiError::FineTune(format!(
                "rank {rank} is not a member rank of a gang of {world} (members are 1..{world})"
            )));
        }
        Ok(Self {
            rank,
            world,
            device,
            timeout: DEFAULT_RENDEZVOUS_TIMEOUT,
            chunk_bytes: chunk_bytes(max_message_bytes)?,
            agreement: None,
            inner: Mutex::new(Inner {
                endpoint: Endpoint::Member(link),
                next_round: 0,
                fault: None,
            }),
        })
    }

    /// The gang deadline every round wait on this rank expires at.
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self> {
        if timeout.is_zero() {
            return Err(JammiError::FineTune(
                "a Peer's round timeout must be > 0 (a zero deadline expires before any peer \
                 can answer)"
                    .into(),
            ));
        }
        self.timeout = timeout;
        Ok(self)
    }

    /// Bind the opaque [`Descriptor::agreement`] digest this rank signs
    /// every round with — see `Local::with_agreement`.
    pub fn with_agreement(mut self, digest: impl Into<String>) -> Self {
        self.agreement = Some(digest.into());
        self
    }

    /// The device this rank trains on and every result lands on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    fn descriptor(
        &self,
        verb: Verb,
        root: Option<u32>,
        counts: Option<Vec<usize>>,
        tensors: Vec<TensorSignature>,
    ) -> Descriptor {
        Descriptor {
            // Stamped by `Peer::round` from this rank's own round counter.
            round: 0,
            verb,
            world: self.world as usize,
            root,
            counts,
            tensors,
            agreement: self.agreement.clone(),
        }
    }

    /// Run one collective inside this rank's fault state: refuse if the
    /// gang has already failed; run `prepare` (the rank's own argument
    /// checks and its contribution), faulting the peers promptly if it
    /// refuses; run one round; `apply` the committed result; and make any
    /// failure permanent.
    fn round<A, T>(
        &self,
        verb: Verb,
        args: A,
        prepare: impl FnOnce(&A) -> Result<(Descriptor, Contribution)>,
        apply: impl FnOnce(A, Pending) -> Result<T>,
    ) -> Result<T> {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(reason) = &inner.fault {
            return Err(JammiError::FineTune(format!(
                "{verb}: the gang has already failed — {reason}. A collective after a failure is \
                 refused rather than run: the peers it would rendezvous with are not coming, and \
                 any result assembled from what they left behind is stale."
            )));
        }
        let round = inner.next_round;
        inner.next_round += 1;
        let Inner {
            endpoint, fault, ..
        } = &mut *inner;

        let (descriptor, contribution) = match prepare(&args) {
            Ok((mut descriptor, contribution)) => {
                descriptor.round = round;
                (descriptor, contribution)
            }
            Err(error) => {
                let reason = format!("{verb}: round {round}: {}", reason_of(&error));
                match endpoint {
                    Endpoint::Coordinator(members) => {
                        for link in members.iter() {
                            link.link.send_fault(round, &reason);
                        }
                    }
                    Endpoint::Member(link) => link.0.send_fault(round, &reason),
                }
                *fault = Some(reason.clone());
                return Err(JammiError::FineTune(reason));
            }
        };

        let deadline = Instant::now() + self.timeout;
        let committed = match endpoint {
            Endpoint::Coordinator(members) => {
                self.coordinate(members, round, verb, &descriptor, contribution, deadline)
            }
            Endpoint::Member(link) => {
                self.participate(link, round, verb, &descriptor, contribution, deadline)
            }
        };
        let result = committed.and_then(|result| apply(args, result));
        if let Err(error) = &result {
            *fault = Some(reason_of(error));
        }
        result
    }

    /// The coordinator's round: collect, agree, fold, publish, ACK, commit.
    fn coordinate(
        &self,
        members: &mut [CoordinatorLink],
        round: u64,
        verb: Verb,
        descriptor: &Descriptor,
        own: Contribution,
        deadline: Instant,
    ) -> Result<Pending> {
        let wire_descriptor = descriptor_to_wire(descriptor).map_err(|reason| {
            fault_all(members, round, format!("{verb}: round {round}: {reason}"))
        })?;

        // Every member's contribution, in rank order, each under a
        // descriptor equal to rank 0's — or no round for anyone.
        let mut contributions: Vec<Contribution> = Vec::with_capacity(members.len() + 1);
        contributions.push(own);
        let collected = members.iter_mut().try_for_each(|link| {
            let rank = link.rank;
            let bound = reassembly_bound(&contribution_shapes(descriptor, rank))
                .map_err(|reason| format!("{verb}: round {round}: {reason}"))?;
            let (payload, body) = recv_payload(
                &mut link.link,
                round,
                verb,
                deadline,
                self.timeout,
                &format!("rank {rank}'s contribution"),
                bound,
            )?;
            let wire = payload
                .descriptor
                .as_ref()
                .expect("recv_payload refuses a payload without a descriptor");
            let theirs = descriptor_from_wire(wire).map_err(|reason| {
                format!(
                    "{verb}: round {round}: rank {rank}'s round descriptor {wire:?} is refused \
                     ({reason}) against rank 0's {descriptor:?} — the ranks disagree about what \
                     this round computes, so no rank may be handed a result"
                )
            })?;
            if !descriptor.agrees_with(&theirs) {
                return Err(format!(
                    "{verb}: round {round}: rank 0's round descriptor is {descriptor:?} but rank \
                     {rank}'s is {theirs:?} — the ranks disagree about what this round computes, \
                     so no rank may be handed a result"
                ));
            }
            let tensors = decode_tensors(&body, &contribution_shapes(descriptor, rank)).map_err(
                |reason| format!("{verb}: round {round}: rank {rank}'s contribution: {reason}"),
            )?;
            contributions.push(match verb {
                Verb::AllGather => Contribution::Gather(single(tensors)),
                Verb::AllReduceSum => Contribution::ReduceSum(tensors),
                Verb::AllReduceMaxFlags => Contribution::MaxFlags(payload.flags),
                Verb::Broadcast => Contribution::Broadcast(tensors.into_iter().next()),
                Verb::Barrier => Contribution::Barrier,
            });
            Ok(())
        });
        if let Err(reason) = collected {
            return Err(fault_all(members, round, reason));
        }

        // The fold, on this rank's device, in rank order.
        let folded = self
            .fold(verb, descriptor, &contributions)
            .map_err(|error| {
                fault_all(
                    members,
                    round,
                    format!("{verb}: round {round}: {}", reason_of(&error)),
                )
            })?;
        let body = Body {
            ipc: encode_tensors(&folded.tensors.iter().collect::<Vec<_>>())?,
            flags: folded.flags,
            chunk_bytes: self.chunk_bytes,
        };
        let published = members.iter().try_for_each(|link| {
            if link
                .link
                .send_payload(round, wire_descriptor.clone(), &body)
            {
                Ok(())
            } else {
                Err(format!(
                    "{verb}: round {round}: rank {} disconnected while the result was being \
                     published — no rank applies round {round}",
                    link.rank
                ))
            }
        });
        if let Err(reason) = published {
            return Err(fault_all(members, round, reason));
        }

        // Every member's ACK, then the commit point.
        let acked = members.iter_mut().try_for_each(|link| {
            let rank = link.rank;
            recv_ack(&mut link.link, round, verb, deadline, self.timeout, rank)
        });
        if let Err(reason) = acked {
            return Err(fault_all(members, round, reason));
        }
        // The commit fan-out. A send the far side did not take is fatal on
        // every rank: this rank applies nothing, every member is faulted
        // (one that already took its commit applied the round and is refused
        // at its next contribution), and the gang is faulted from here on.
        let fanned_out = members.iter().try_for_each(|link| {
            if link.link.send(RankControl {
                control: Some(rank_control::Control::RoundCommit(RoundCommit { round })),
            }) {
                Ok(())
            } else {
                Err(format!(
                    "{verb}: round {round}: rank {} disconnected during the commit fan-out — the \
                     coordinator applies nothing, and the gang is faulted for every rank",
                    link.rank
                ))
            }
        });
        if let Err(reason) = fanned_out {
            return Err(fault_all(members, round, reason));
        }
        Ok(folded)
    }

    /// A member's round: contribute, wait, hold, ACK, wait, apply.
    fn participate(
        &self,
        link: &mut MemberLink,
        round: u64,
        verb: Verb,
        descriptor: &Descriptor,
        own: Contribution,
        deadline: Instant,
    ) -> Result<Pending> {
        let link = &mut link.0;
        let wire_descriptor = descriptor_to_wire(descriptor).map_err(|reason| {
            let reason = format!("{verb}: round {round}: {reason}");
            link.send_fault(round, &reason);
            JammiError::FineTune(reason)
        })?;
        let body = Body {
            ipc: encode_tensors(&own.tensors())?,
            flags: own.flags(),
            chunk_bytes: self.chunk_bytes,
        };
        if !link.send_payload(round, wire_descriptor, &body) {
            return Err(JammiError::FineTune(format!(
                "{verb}: round {round}: the stream to the coordinator ended while this rank's \
                 contribution was being sent — nothing applied"
            )));
        }

        // The coordinator's result: its descriptor must be the one this rank
        // contributed under (defence in depth — the coordinator already
        // refused any disagreement before publishing).
        let shapes = result_shapes(descriptor);
        let bound = reassembly_bound(&shapes).map_err(|reason| {
            let reason = format!("{verb}: round {round}: {reason}");
            link.send_fault(round, &reason);
            JammiError::FineTune(reason)
        })?;
        let (payload, body) = recv_payload(
            link,
            round,
            verb,
            deadline,
            self.timeout,
            "the coordinator's result",
            bound,
        )
        .map_err(|reason| {
            link.send_fault(round, &reason);
            JammiError::FineTune(reason)
        })?;
        let wire = payload
            .descriptor
            .as_ref()
            .expect("recv_payload refuses a payload without a descriptor");
        let theirs = descriptor_from_wire(wire).map_err(|reason| {
            let reason = format!(
                "{verb}: round {round}: the coordinator's round descriptor {wire:?} is refused \
                 ({reason}) against this rank's {descriptor:?}"
            );
            link.send_fault(round, &reason);
            JammiError::FineTune(reason)
        })?;
        if !descriptor.agrees_with(&theirs) {
            let reason = format!(
                "{verb}: round {round}: the coordinator published under {theirs:?} but this \
                 rank {} contributed under {descriptor:?} — refused",
                self.rank
            );
            link.send_fault(round, &reason);
            return Err(JammiError::FineTune(reason));
        }
        let pending = decode_tensors(&body, &shapes)
            .map(|tensors| Pending {
                tensors,
                flags: payload.flags,
            })
            .map_err(|reason| {
                let reason = format!("{verb}: round {round}: the coordinator's result: {reason}");
                link.send_fault(round, &reason);
                JammiError::FineTune(reason)
            })?;

        // Held, unapplied. ACK, then wait for the commit.
        if !link.send(RankEvent {
            event: Some(rank_event::Event::RoundAck(RoundAck { round })),
        }) {
            return Err(JammiError::FineTune(format!(
                "{verb}: round {round}: the stream to the coordinator ended before this rank's \
                 ACK was sent — nothing applied"
            )));
        }
        recv_commit(link, round, verb, deadline, self.timeout).map_err(|reason| {
            link.send_fault(round, &reason);
            JammiError::FineTune(reason)
        })?;
        Ok(pending)
    }

    /// The fold, on this rank's device, in rank order — the same operation
    /// sequence [`Local`](super::Local) runs on rank 0's device.
    fn fold(
        &self,
        verb: Verb,
        descriptor: &Descriptor,
        contributions: &[Contribution],
    ) -> Result<Pending> {
        match verb {
            Verb::AllGather => {
                let mut slices = Vec::with_capacity(contributions.len());
                for contribution in contributions {
                    let Contribution::Gather(tensor) = contribution else {
                        unreachable!("every contribution of a round carries the round's verb");
                    };
                    if tensor.dims().first().copied().unwrap_or(0) == 0 {
                        continue;
                    }
                    slices.push(tensor.to_device(&self.device).map_err(|e| {
                        JammiError::FineTune(format!("all_gather: to_device: {e}"))
                    })?);
                }
                let gathered = if slices.is_empty() {
                    // Every rank contributed zero rows: rank 0's own (empty)
                    // tensor already carries the trailing shape and dtype.
                    let Contribution::Gather(own) = &contributions[0] else {
                        unreachable!("rank 0's contribution carries the round's verb");
                    };
                    own.clone()
                } else if slices.len() == 1 {
                    slices.into_iter().next().expect("length checked")
                } else {
                    Tensor::cat(&slices, 0)
                        .map_err(|e| JammiError::FineTune(format!("all_gather: cat: {e}")))?
                };
                Ok(Pending {
                    tensors: vec![gathered],
                    flags: 0,
                })
            }
            Verb::AllReduceSum => {
                let count = descriptor.tensors.len();
                let mut sums = Vec::with_capacity(count);
                for index in 0..count {
                    let mut sum: Option<Tensor> = None;
                    for (peer, contribution) in contributions.iter().enumerate() {
                        let Contribution::ReduceSum(tensors) = contribution else {
                            unreachable!("every contribution of a round carries the round's verb");
                        };
                        let term = tensors[index].to_device(&self.device).map_err(|e| {
                            JammiError::FineTune(format!("all_reduce_sum: to_device: {e}"))
                        })?;
                        sum = Some(match sum {
                            None => term,
                            Some(acc) => acc.add(&term).map_err(|e| {
                                JammiError::FineTune(format!(
                                    "all_reduce_sum: adding rank {peer}'s tensor {index}: {e}"
                                ))
                            })?,
                        });
                    }
                    sums.push(sum.expect("a gang has at least one rank"));
                }
                Ok(Pending {
                    tensors: sums,
                    flags: 0,
                })
            }
            Verb::AllReduceMaxFlags => {
                let mut max = 0u32;
                for contribution in contributions {
                    let Contribution::MaxFlags(flags) = contribution else {
                        unreachable!("every contribution of a round carries the round's verb");
                    };
                    max = max.max(*flags);
                }
                Ok(Pending {
                    tensors: Vec::new(),
                    flags: max,
                })
            }
            Verb::Broadcast => {
                let root = descriptor
                    .root
                    .expect("a broadcast descriptor names its root")
                    as usize;
                let Contribution::Broadcast(from_root) = &contributions[root] else {
                    unreachable!("every contribution of a round carries the round's verb");
                };
                let from_root = from_root
                    .as_ref()
                    .ok_or_else(|| {
                        JammiError::FineTune(format!(
                            "broadcast: rank {root} is the agreed root but sent no tensor"
                        ))
                    })?
                    .to_device(&self.device)
                    .map_err(|e| JammiError::FineTune(format!("broadcast: to_device: {e}")))?;
                Ok(Pending {
                    tensors: vec![from_root],
                    flags: 0,
                })
            }
            Verb::Barrier => Ok(Pending {
                tensors: Vec::new(),
                flags: 0,
            }),
        }
    }
}

fn single(mut tensors: Vec<Tensor>) -> Tensor {
    tensors
        .pop()
        .expect("a one-signature contribution decodes to exactly one tensor")
}

/// The chunk payload size under a listener cap.
fn chunk_bytes(max_message_bytes: usize) -> Result<usize> {
    if max_message_bytes < MIN_MESSAGE_CAP {
        return Err(JammiError::FineTune(format!(
            "a gang message cap of {max_message_bytes} bytes is below the {MIN_MESSAGE_CAP}-byte \
             minimum a chunk frame needs"
        )));
    }
    Ok(max_message_bytes - CHUNK_FRAME_OVERHEAD)
}

/// Fault every member and hand the coordinator its error.
fn fault_all(members: &[CoordinatorLink], round: u64, reason: String) -> JammiError {
    for link in members {
        link.link.send_fault(round, &reason);
    }
    JammiError::FineTune(reason)
}

/// The words to record as a gang's fault for `error` (a
/// [`JammiError::FineTune`]'s own message rather than its `Display`, so a
/// fault quoted inside a later refusal does not carry a second prefix).
fn reason_of(error: &JammiError) -> String {
    match error {
        JammiError::FineTune(message) => message.clone(),
        other => other.to_string(),
    }
}

/// Describe a wait that ended without the frame it waited for.
fn wait_failed(end: WaitEnd, verb: Verb, round: u64, timeout: Duration, what: &str) -> String {
    match end {
        WaitEnd::Timeout => format!(
            "{verb}: round {round}: timed out after {timeout:?} waiting for {what} — nothing \
             applied"
        ),
        WaitEnd::Disconnected => format!(
            "{verb}: round {round}: the stream ended while waiting for {what} — nothing applied"
        ),
        WaitEnd::Transport(reason) => format!(
            "{verb}: round {round}: the stream failed ({reason}) while waiting for {what} — \
             nothing applied"
        ),
    }
}

/// A payload for `round` and its chunks, reassembled under `bound`. Any
/// other frame for this round, a frame for another round, a fault, or a
/// session frame ends the wait with the reason.
fn recv_payload<In: Inbound, Out: Outbound>(
    link: &mut Link<In, Out>,
    round: u64,
    verb: Verb,
    deadline: Instant,
    timeout: Duration,
    what: &str,
    bound: usize,
) -> std::result::Result<(RoundPayload, Vec<u8>), String> {
    let payload = match link
        .recv(deadline)
        .map_err(|end| wait_failed(end, verb, round, timeout, what))?
    {
        Frame::Payload(payload) if payload.round == round => payload,
        Frame::Payload(payload) => {
            return Err(format!(
                "{verb}: round {round}: a payload for round {} arrived where {what} for round \
                 {round} was expected — the ranks are not in lockstep",
                payload.round
            ));
        }
        Frame::Fault(fault) => {
            return Err(format!(
                "{verb}: round {round}: faulted by a peer — {}",
                fault.detail
            ));
        }
        other => return Err(unexpected(other, verb, round, what)),
    };
    if payload.descriptor.is_none() {
        return Err(format!(
            "{verb}: round {round}: {what} carries no descriptor"
        ));
    }
    let chunk_count = payload.chunk_count as usize;
    let mut ipc = Vec::new();
    for index in 0..chunk_count {
        let chunk = match link.recv(deadline).map_err(|end| {
            wait_failed(
                end,
                verb,
                round,
                timeout,
                &format!("chunk {index} of {what}"),
            )
        })? {
            Frame::Chunk(chunk) if chunk.round == round && chunk.index as usize == index => chunk,
            Frame::Chunk(chunk) => {
                return Err(format!(
                    "{verb}: round {round}: chunk {} of round {} arrived where chunk {index} of \
                     round {round} was expected",
                    chunk.index, chunk.round
                ));
            }
            Frame::Fault(fault) => {
                return Err(format!(
                    "{verb}: round {round}: faulted by a peer — {}",
                    fault.detail
                ));
            }
            other => return Err(unexpected(other, verb, round, what)),
        };
        if ipc.len() + chunk.ipc.len() > bound {
            return Err(format!(
                "{verb}: round {round}: {what} exceeds the {bound}-byte bound its descriptor \
                 implies at chunk {index} — refused before buffering more"
            ));
        }
        ipc.extend_from_slice(&chunk.ipc);
    }
    Ok((payload, ipc))
}

fn recv_ack<In: Inbound, Out: Outbound>(
    link: &mut Link<In, Out>,
    round: u64,
    verb: Verb,
    deadline: Instant,
    timeout: Duration,
    rank: u32,
) -> std::result::Result<(), String> {
    let what = format!("rank {rank}'s ACK");
    match link
        .recv(deadline)
        .map_err(|end| wait_failed(end, verb, round, timeout, &what))
        .map_err(|reason| format!("{reason}; no rank applies round {round}"))?
    {
        Frame::Ack(ack) if ack.round == round => Ok(()),
        Frame::Ack(ack) => Err(format!(
            "{verb}: round {round}: rank {rank} ACKed round {} where round {round} was \
             expected — no rank applies round {round}",
            ack.round
        )),
        Frame::Fault(fault) => Err(format!(
            "{verb}: round {round}: rank {rank} faulted before its ACK — {} — no rank applies \
             round {round}",
            fault.detail
        )),
        other => Err(unexpected(other, verb, round, &what)),
    }
}

fn recv_commit<In: Inbound, Out: Outbound>(
    link: &mut Link<In, Out>,
    round: u64,
    verb: Verb,
    deadline: Instant,
    timeout: Duration,
) -> std::result::Result<(), String> {
    let what = "the coordinator's commit";
    match link
        .recv(deadline)
        .map_err(|end| wait_failed(end, verb, round, timeout, what))?
    {
        Frame::Commit(commit) if commit.round == round => Ok(()),
        Frame::Commit(commit) => Err(format!(
            "{verb}: round {round}: a commit for round {} arrived where round {round}'s was \
             expected — nothing applied",
            commit.round
        )),
        Frame::Fault(fault) => Err(format!(
            "{verb}: round {round}: faulted by a peer before the commit — {} — nothing applied",
            fault.detail
        )),
        other => Err(unexpected(other, verb, round, what)),
    }
}

fn unexpected(frame: Frame, verb: Verb, round: u64, what: &str) -> String {
    match frame {
        Frame::Session(description) => format!(
            "{verb}: round {round}: {description} while waiting for {what} — nothing applied"
        ),
        other => format!(
            "{verb}: round {round}: {other:?} arrived while waiting for {what} — the ranks are \
             not in lockstep, nothing applied"
        ),
    }
}

// ── Collective ──────────────────────────────────────────────────────────────

impl Collective for Peer {
    fn all_gather(&self, _call: &BlockingCall, local: &Tensor, counts: &[usize]) -> Result<Tensor> {
        let rank = self.rank as usize;
        self.round(
            Verb::AllGather,
            (),
            |()| {
                checked_gather_counts(self.rank, self.world, local, counts)?;
                Ok((
                    self.descriptor(
                        Verb::AllGather,
                        None,
                        Some(counts.to_vec()),
                        vec![TensorSignature::of_gather_slice(local)],
                    ),
                    Contribution::Gather(local.clone()),
                ))
            },
            |(), result| {
                let total: usize = counts.iter().sum();
                let own_rows = counts[rank];
                let rows_before: usize = counts[..rank].iter().sum();
                let gathered = single(result.tensors)
                    .to_device(&self.device)
                    .map_err(|e| JammiError::FineTune(format!("all_gather: to_device: {e}")))?;
                let rows = gathered.dims().first().copied().unwrap_or(0);
                if rows != total {
                    return Err(JammiError::FineTune(format!(
                        "all_gather: gathered {rows} rows where the counts sum to {total}"
                    )));
                }
                // Only this rank's own slot carries a gradient: the remote
                // slots are the published values, detached; the caller's own
                // `local` is spliced back in at its rank's rows, byte-equal
                // to what the coordinator folded there.
                if own_rows == 0 {
                    return Ok(gathered.detach());
                }
                if rows == own_rows {
                    return Ok(local.clone());
                }
                let mut parts = Vec::with_capacity(3);
                if rows_before > 0 {
                    parts.push(
                        gathered
                            .narrow(0, 0, rows_before)
                            .map_err(|e| JammiError::FineTune(format!("all_gather: narrow: {e}")))?
                            .detach(),
                    );
                }
                parts.push(local.clone());
                let after = rows - rows_before - own_rows;
                if after > 0 {
                    parts.push(
                        gathered
                            .narrow(0, rows_before + own_rows, after)
                            .map_err(|e| JammiError::FineTune(format!("all_gather: narrow: {e}")))?
                            .detach(),
                    );
                }
                Tensor::cat(&parts, 0)
                    .map_err(|e| JammiError::FineTune(format!("all_gather: cat: {e}")))
            },
        )
    }

    fn all_reduce_sum(&self, _call: &BlockingCall, tensors: &mut [Tensor]) -> Result<()> {
        self.round(
            Verb::AllReduceSum,
            tensors,
            |tensors| {
                Ok((
                    self.descriptor(
                        Verb::AllReduceSum,
                        None,
                        None,
                        tensors.iter().map(TensorSignature::of).collect(),
                    ),
                    Contribution::ReduceSum(tensors.to_vec()),
                ))
            },
            |tensors, result| {
                for (slot, sum) in tensors.iter_mut().zip(result.tensors) {
                    *slot = sum.to_device(&self.device).map_err(|e| {
                        JammiError::FineTune(format!("all_reduce_sum: to_device: {e}"))
                    })?;
                }
                Ok(())
            },
        )
    }

    fn all_reduce_max_flags(&self, _call: &BlockingCall, flags: u32) -> Result<u32> {
        self.round(
            Verb::AllReduceMaxFlags,
            (),
            |()| {
                Ok((
                    self.descriptor(Verb::AllReduceMaxFlags, None, None, Vec::new()),
                    Contribution::MaxFlags(flags),
                ))
            },
            |(), result| Ok(result.flags),
        )
    }

    fn broadcast(&self, _call: &BlockingCall, t: &mut Tensor, root: u32) -> Result<()> {
        self.round(
            Verb::Broadcast,
            t,
            |t| {
                checked_root(self.world, root)?;
                Ok((
                    self.descriptor(
                        Verb::Broadcast,
                        Some(root),
                        None,
                        vec![TensorSignature::of(t)],
                    ),
                    Contribution::Broadcast((self.rank == root).then(|| Tensor::clone(t))),
                ))
            },
            |t, result| {
                *t = single(result.tensors)
                    .to_device(&self.device)
                    .map_err(|e| JammiError::FineTune(format!("broadcast: to_device: {e}")))?
                    .detach();
                Ok(())
            },
        )
    }

    fn barrier(&self, _call: &BlockingCall) -> Result<()> {
        self.round(
            Verb::Barrier,
            (),
            |()| {
                Ok((
                    self.descriptor(Verb::Barrier, None, None, Vec::new()),
                    Contribution::Barrier,
                ))
            },
            |(), _| Ok(()),
        )
    }

    fn rank(&self) -> u32 {
        self.rank
    }

    fn world(&self) -> u32 {
        self.world
    }
}

// ── The rank's read path ────────────────────────────────────────────────────

/// A fault a gang rank raises on ITS OWN read path, before its first
/// collective. Every arm is MEMBER-scoped by construction: the assembly-
/// scoped class (a refuted row fact) is not constructible here, because
/// nothing on the read path decides one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RankReadFault {
    /// This host's object store did not hand back the attested bytes of a
    /// leaf: a read error, a short read, or a digest that is not the
    /// inventory's. `key` names the leaf when one was being read. Mapped to
    /// the wire's `ABORT_REASON_STORE_UNAVAILABLE`; never counted against
    /// the assembly's attempt budget.
    StoreUnavailable {
        key: Option<LeafKey>,
        detail: String,
    },
}

impl RankReadFault {
    /// The frozen wire reason this fault ends a session with.
    pub fn abort_reason(&self) -> AbortReason {
        match self {
            Self::StoreUnavailable { .. } => AbortReason::StoreUnavailable,
        }
    }
}

impl fmt::Display for RankReadFault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StoreUnavailable { key, detail } => match key {
                Some(key) => write!(
                    f,
                    "store unavailable (member-scoped) at leaf {key:?}: {detail}"
                ),
                None => write!(f, "store unavailable (member-scoped): {detail}"),
            },
        }
    }
}

impl std::error::Error for RankReadFault {}

/// The byte range one leaf spans, or the fault for a leaf this reader
/// cannot locate in a Parquet object.
fn leaf_range(leaf: &LeafDigest) -> std::result::Result<Range<u64>, RankReadFault> {
    match &leaf.key {
        LeafKey::RowGroup { offset, length, .. } => {
            let end =
                offset
                    .checked_add(*length)
                    .ok_or_else(|| RankReadFault::StoreUnavailable {
                        key: Some(leaf.key.clone()),
                        detail: "the leaf's byte range overflows".into(),
                    })?;
            Ok(*offset..end)
        }
        LeafKey::File { name } => Err(RankReadFault::StoreUnavailable {
            key: Some(leaf.key.clone()),
            detail: format!(
                "the inventory names a bundle file '{name}' where a Parquet row group was \
                 expected — this reader cannot locate it"
            ),
        }),
    }
}

/// `bytes` must be exactly the leaf's range and hash to its digest.
fn check_leaf(
    leaf: &LeafDigest,
    range: &Range<u64>,
    bytes: &[u8],
) -> std::result::Result<(), RankReadFault> {
    let expected_len = range.end - range.start;
    if bytes.len() as u64 != expected_len {
        return Err(RankReadFault::StoreUnavailable {
            key: Some(leaf.key.clone()),
            detail: format!(
                "a short read: {} bytes where the leaf spans {expected_len}",
                bytes.len()
            ),
        });
    }
    let found = ArtifactDigest::of_bytes(bytes);
    if found != leaf.digest {
        return Err(RankReadFault::StoreUnavailable {
            key: Some(leaf.key.clone()),
            detail: format!(
                "digest {} where the inventory attests {}",
                found.0, leaf.digest.0
            ),
        });
    }
    Ok(())
}

/// Verify `leaves` one at a time through `read`, which hands back exactly
/// the bytes of one leaf's range — never wider. Stops at the first leaf
/// whose bytes are not the attested ones, so memory is bounded by the
/// largest leaf, and the fault names it.
pub fn verify_leaves(
    leaves: &[LeafDigest],
    mut read: impl FnMut(&LeafKey, Range<u64>) -> std::result::Result<Bytes, String>,
) -> std::result::Result<(), RankReadFault> {
    for leaf in leaves {
        let range = leaf_range(leaf)?;
        let bytes =
            read(&leaf.key, range.clone()).map_err(|detail| RankReadFault::StoreUnavailable {
                key: Some(leaf.key.clone()),
                detail,
            })?;
        check_leaf(leaf, &range, &bytes)?;
    }
    Ok(())
}

/// [`verify_leaves`] over the object `handle` was opened against: one
/// ranged read per leaf, never the whole object.
pub async fn verify_partition_leaves(
    handle: &JammiObjectStore,
    leaves: &[LeafDigest],
) -> std::result::Result<(), RankReadFault> {
    let path = handle
        .data_path()
        .map_err(|e| RankReadFault::StoreUnavailable {
            key: None,
            detail: e.to_string(),
        })?;
    for leaf in leaves {
        let range = leaf_range(leaf)?;
        let bytes = handle.get_range(&path, range.clone()).await.map_err(|e| {
            RankReadFault::StoreUnavailable {
                key: Some(leaf.key.clone()),
                detail: e.to_string(),
            }
        })?;
        check_leaf(leaf, &range, &bytes)?;
    }
    Ok(())
}
