//! Consumer-offset snapshot returned by
//! [`crate::trigger::broker::TriggerBroker::list_consumers`].
//!
//! The shape is what backup/restore needs to capture and re-apply on a fresh
//! broker instance: a stable identifier per consumer, the topic it is bound
//! to, and the two ENGINE-OFFSET positions every driver tracks (the last
//! engine offset delivered to the consumer, and the ack floor below which
//! every event has been explicitly acknowledged) — never a driver-native
//! sequence. Fields are `Option<u64>` because a driver may not be able to
//! resolve an engine offset for a consumer (the Postgres wake-up driver has
//! seen no NOTIFY yet) — `None` then, never a fabricated `0`.
//!
//! The in-memory broker does not implement explicit acks — its
//! `last_acked_offset` equals `last_delivered_offset` so a backup-restore
//! cycle through it round-trips the same value.

use crate::trigger::ids::TopicId;

/// One consumer's offset state at the moment [`list_consumers`] was
/// called. It holds only the fields the engine's backup/restore path
/// consumes, so no driver-specific bookkeeping leaks through the surface.
///
/// Both fields are **engine `_offset`s**, not a driver-native sequence, for
/// every driver: a restore primes a fresh broker via
/// `subscribe(from_offset = engine offset)`, and a driver-native sequence is
/// meaningless across drivers. `None` means this driver could not resolve an
/// engine offset for the consumer — never fabricated as `0`.
///
/// [`list_consumers`]: crate::trigger::broker::TriggerBroker::list_consumers
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsumerOffsetSnapshot {
    /// Broker-assigned consumer identifier — the UUID of the
    /// [`crate::trigger::ids::SubscriptionId`] that opened the subscription.
    /// Stable across `list_consumers` calls for the lifetime of the consumer.
    pub consumer_name: String,
    /// Topic the consumer is bound to.
    pub topic_id: TopicId,
    /// The engine `_offset` of the last event delivered to this consumer.
    pub last_delivered_offset: Option<u64>,
    /// The engine `_offset` below which every event has been acknowledged.
    /// For brokers without an ack model (every shipped driver), this equals
    /// `last_delivered_offset`.
    pub last_acked_offset: Option<u64>,
}
