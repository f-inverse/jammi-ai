//! Subscription handle returned by [`crate::trigger::broker::TriggerBroker::subscribe`].

use std::pin::Pin;
use std::task::{Context, Poll};

use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Utc};
use futures::Stream;

use crate::tenant::TenantId;
use crate::trigger::error::TriggerError;
use crate::trigger::ids::SubscriptionId;
use crate::trigger::offset::Offset;

/// One batch delivered through a subscription. The broker assigns `offset`
/// at publish time; `produced_at` is the same instant as `offset.committed_at`
/// — surfaced as a top-level field so consumers do not have to reach into the
/// offset object for the common case.
///
/// `tenant` is the publish-scoped tenant tag the broker carried opaquely from
/// [`crate::trigger::broker::TriggerBroker::publish`] through to this
/// delivery — the engine's live-tail subscribe seam
/// ([`crate::trigger::Subscriber::subscribe_scoped`]) filters on it. It is
/// OUT-OF-BAND metadata only: no wire encoder may put it on the wire (the
/// `jammi-wire` encoder carries `batch` / `offset` / `produced_at` only), and
/// a `DeliveredBatch` reconstructed from a decoded wire frame carries
/// `tenant: None` because the filtering already happened server-side before
/// the frame was sent.
#[derive(Debug, Clone)]
pub struct DeliveredBatch {
    pub offset: Offset,
    pub produced_at: DateTime<Utc>,
    pub batch: RecordBatch,
    pub tenant: Option<TenantId>,
}

/// A `Stream` of delivered batches. Drop to unsubscribe — the driver's
/// underlying handle tears down through normal `Drop` propagation on the
/// pinned inner stream.
pub struct Subscription {
    pub id: SubscriptionId,
    inner: Pin<Box<dyn Stream<Item = Result<DeliveredBatch, TriggerError>> + Send + 'static>>,
}

impl Subscription {
    pub fn new(
        id: SubscriptionId,
        inner: Pin<Box<dyn Stream<Item = Result<DeliveredBatch, TriggerError>> + Send + 'static>>,
    ) -> Self {
        Self { id, inner }
    }
}

impl Stream for Subscription {
    type Item = Result<DeliveredBatch, TriggerError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        this.inner.as_mut().poll_next(cx)
    }
}

/// One item a [`crate::trigger::broker::TriggerBroker`] driver hands the
/// engine's subscribe seam ([`crate::trigger::Subscriber`]).
///
/// A driver that carries the published bytes itself (the in-memory broker,
/// JetStream) yields [`LiveEvent::Batch`]. A driver that carries no payload
/// — a wake-up transport over the authoritative backing table — yields only
/// [`LiveEvent::Wake`]. `Wake` is payload-free: it means "the backing table
/// MAY have new rows past what this subscriber has already seen", never a
/// specific offset bound, so the engine always replays to the current head
/// rather than trusting a driver-carried upper bound.
///
/// `TriggerError::OffsetEvicted` no longer exists:
/// a driver that cannot deliver at or before `from_offset` — or a lagging
/// receiver that dropped events — yields `Wake` instead of failing the
/// stream. The engine's subscribe seam self-heals by replaying the backing
/// table; the driver's own history depth is never a caller-visible error.
#[derive(Debug, Clone)]
pub enum LiveEvent {
    /// A batch the driver delivered directly.
    Batch(DeliveredBatch),
    /// "The backing table may have advanced past what you've already seen —
    /// replay to find out." Carries no offset.
    Wake,
}

/// A `Stream` of driver-level [`LiveEvent`]s, returned by
/// [`crate::trigger::broker::TriggerBroker::subscribe`]. Distinct from
/// [`Subscription`] (whose item is a plain [`DeliveredBatch`]): `LiveStream`
/// is the driver-facing type, `Subscription` is the engine-facing type
/// [`crate::trigger::Subscriber::subscribe_scoped`] returns after resolving
/// every `Wake` into replayed rows. Drop to unsubscribe, same as
/// `Subscription`.
pub struct LiveStream {
    pub id: SubscriptionId,
    inner: Pin<Box<dyn Stream<Item = Result<LiveEvent, TriggerError>> + Send + 'static>>,
}

impl LiveStream {
    pub fn new(
        id: SubscriptionId,
        inner: Pin<Box<dyn Stream<Item = Result<LiveEvent, TriggerError>> + Send + 'static>>,
    ) -> Self {
        Self { id, inner }
    }
}

impl Stream for LiveStream {
    type Item = Result<LiveEvent, TriggerError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        this.inner.as_mut().poll_next(cx)
    }
}
