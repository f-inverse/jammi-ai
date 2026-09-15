//! The round protocol's seam in this crate: what an ADMITTED session's hold
//! loop hands a round frame to, and how a coordinator dials a member.
//!
//! The round machinery itself is the engine's
//! (`jammi_ai::fine_tune::collective::peer` — the `Peer` collective, its
//! two-phase round, its links). This module owns only the two edges that
//! touch this crate's stream handling:
//!
//! - **The member side.** `GangServer::run_rank` admits a stream and hands
//!   it to a hold loop that owns the inbound `Streaming<RankControl>` (its
//!   `select!` reads it alongside re-verification and drain). That loop
//!   calls [`RoundInbox::deliver`] on every frame [`RoundInbox::is_round_frame`]
//!   recognises, and the member's `Peer` — built over the [`MemberLink`]
//!   [`member_link`] returns beside the inbox — reads them from its link.
//!   The link's outbound events ride the session's own response stream.
//! - **The coordinator side.** [`dial_member`] opens `RunRank` on a
//!   member's `peer_bind` listener with the coordinator's `Assign`, requires
//!   `Admitted`, and returns the [`CoordinatorLink`] the coordinator's `Peer`
//!   is built from — with the client's inbound decode capped at the SAME
//!   `[server.limits] max_message_bytes` every listener applies.

use jammi_ai::fine_tune::collective::{CoordinatorLink, LinkFault, MemberLink};
use jammi_db::catalog::instance::PeerAddr;
use jammi_db::error::{JammiError, Result};
use jammi_wire::proto::gang::{rank_control, Assign, RankControl, RankEvent};
use tokio::sync::mpsc;
use tonic::transport::Channel;
use tonic::Status;

/// Frames buffered between the hold loop and the member's `Peer` before
/// [`RoundInbox::deliver`] waits.
const INBOX_CAPACITY: usize = 64;

/// Where an admitted session's hold loop delivers the round frames it reads
/// off the inbound stream. Dropping it is how the loop tells the member's
/// `Peer` the stream ended.
#[derive(Debug)]
pub struct RoundInbox {
    frames: mpsc::Sender<std::result::Result<RankControl, LinkFault>>,
}

impl RoundInbox {
    /// `true` for a frame the round protocol owns — one of the `round_*`
    /// arms. Every other arm (`Assign`, `Cancel`) is the session's, decided
    /// by the hold loop itself and never delivered here.
    pub fn is_round_frame(frame: &RankControl) -> bool {
        matches!(
            frame.control,
            Some(
                rank_control::Control::RoundResult(_)
                    | rank_control::Control::RoundChunk(_)
                    | rank_control::Control::RoundCommit(_)
                    | rank_control::Control::RoundFault(_)
            )
        )
    }

    /// Hand one round frame to the member's link. `false` once the member's
    /// `Peer` is gone (its round ended in a fault, or the rank body
    /// returned), which the hold loop treats as the end of the session's
    /// round traffic.
    pub async fn deliver(&self, frame: RankControl) -> bool {
        self.frames.send(Ok(frame)).await.is_ok()
    }

    /// Tell the member's link the inbound stream FAILED (a transport error
    /// the hold loop read), so the round in progress ends with that reason
    /// rather than a bare close.
    pub async fn fail(&self, status: &Status) -> bool {
        self.frames
            .send(Err(LinkFault(status.to_string())))
            .await
            .is_ok()
    }
}

/// The member's link for an admitted session, beside the inbox its hold
/// loop delivers round frames to. `events` is the session's own outbound
/// event sender (the response stream's), so the link's `RankEvent`s ride
/// the same stream the session's `Aborted`/`Outcome` do. Must be called
/// inside a runtime context (the handler's).
pub fn member_link(
    events: mpsc::Sender<std::result::Result<RankEvent, Status>>,
) -> Result<(RoundInbox, MemberLink)> {
    let (frames, inbound) = mpsc::channel(INBOX_CAPACITY);
    let (outbound, mut link_events) = mpsc::channel::<RankEvent>(INBOX_CAPACITY);
    // The link speaks bare events; the session's stream carries statuses.
    tokio::spawn(async move {
        while let Some(event) = link_events.recv().await {
            if events.send(Ok(event)).await.is_err() {
                break;
            }
        }
    });
    let link = MemberLink::from_channels(inbound, outbound)?;
    Ok((RoundInbox { frames }, link))
}

/// Open `RunRank` on the member at `addr` (its `peer_bind` listener) with
/// the coordinator's `assign`, require `Admitted`, and return the link the
/// coordinator's `Peer` is built from. The client's inbound decode is
/// capped at `max_message_bytes` — the SAME `[server.limits]` value the
/// member's listener applies to what this coordinator sends.
pub async fn dial_member(
    addr: &PeerAddr,
    assign: Assign,
    max_message_bytes: usize,
) -> Result<CoordinatorLink> {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .map_err(|e| JammiError::FineTune(format!("dial {addr}: {e}")))?
        .connect()
        .await
        .map_err(|e| JammiError::FineTune(format!("dial {addr}: {e}")))?;
    CoordinatorLink::over_client(channel, assign, max_message_bytes).await
}
