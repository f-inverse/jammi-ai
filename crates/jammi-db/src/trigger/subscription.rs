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
