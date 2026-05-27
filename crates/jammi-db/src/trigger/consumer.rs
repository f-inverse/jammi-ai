//! Consumer-offset snapshot returned by
//! [`crate::trigger::broker::TriggerBroker::list_consumers`].
//!
//! The shape mirrors the subset of `async_nats::jetstream::consumer::Info`
//! that backup/restore needs to capture and re-apply on a fresh broker
//! instance: a stable identifier per consumer, the topic it is bound to,
//! and the two stream-sequence positions JetStream tracks (the last
//! sequence delivered to the consumer, and the ack floor below which
//! every message has been explicitly acknowledged). Fields are `u64`
//! because JetStream's sequence numbers are `u64`; truncating to `i64`
//! would be a silent precision loss at the API boundary.
//!
//! The in-memory broker does not implement explicit acks — its
//! `last_ack_stream_sequence` equals `last_delivered_stream_sequence` so
//! a backup-restore cycle through the mock round-trips the same value.

use crate::trigger::ids::TopicId;

/// One consumer's offset state at the moment [`list_consumers`] was
/// called. JetStream calls this its `consumer::Info`; we keep only the
/// fields that the engine's backup/restore path consumes so the OSS
/// surface does not leak driver-specific bookkeeping (cluster info,
/// pause state, push-bound flags, …).
///
/// [`list_consumers`]: crate::trigger::broker::TriggerBroker::list_consumers
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsumerOffsetSnapshot {
    /// Broker-assigned consumer identifier. For JetStream this is the
    /// consumer `name` returned by `consumer::Info`; for the in-memory
    /// broker it is the UUID of the [`crate::trigger::ids::SubscriptionId`]
    /// that opened the subscription. Stable across `list_consumers`
    /// calls for the lifetime of the consumer.
    pub consumer_name: String,
    /// Topic the consumer is bound to.
    pub topic_id: TopicId,
    /// The stream-sequence number of the last message the broker has
    /// delivered to this consumer. Equivalent to JetStream's
    /// `Info.delivered.stream_sequence`.
    pub last_delivered_stream_sequence: u64,
    /// The stream-sequence below which every message has been
    /// acknowledged. Equivalent to JetStream's
    /// `Info.ack_floor.stream_sequence`. For brokers without an ack
    /// model (the in-memory broker), this equals
    /// `last_delivered_stream_sequence`.
    pub last_ack_stream_sequence: u64,
}
