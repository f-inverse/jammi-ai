//! `TriggerBroker` trait — pluggable transport for trigger-stream topics.

use arrow::record_batch::RecordBatch;
use async_trait::async_trait;

use crate::trigger::consumer::ConsumerOffsetSnapshot;
use crate::trigger::error::TriggerError;
use crate::trigger::ids::TopicId;
use crate::trigger::offset::Offset;
use crate::trigger::predicate::Predicate;
use crate::trigger::subscription::Subscription;
use crate::trigger::topic::TopicDefinition;

/// A pluggable pub/sub backend. Implementations are responsible only for
/// *transport* — fan-out from publisher to live subscribers. Persistence is
/// the engine's concern via the Phase-2 backing table; the broker never
/// sees `tenant_id` (catalog lookup and the engine's predicate injection
/// enforce tenant scope upstream).
#[async_trait]
pub trait TriggerBroker: Send + Sync + 'static {
    /// Idempotently register a topic. Re-registering an existing topic
    /// with the same schema is a no-op; a schema mismatch returns
    /// [`TriggerError::SchemaConflict`].
    async fn register_topic(&self, topic: &TopicDefinition) -> Result<(), TriggerError>;

    /// Drop a topic from the driver. The backing table is the engine's
    /// concern and is dropped separately.
    async fn drop_topic(&self, topic_id: TopicId) -> Result<(), TriggerError>;

    /// Fan out a batch to currently-attached subscribers. Returns the
    /// offset the driver assigned. MUST NOT persist — the backing table
    /// is the engine's authoritative log.
    async fn publish(
        &self,
        topic_id: TopicId,
        batch: RecordBatch,
        produced_at: chrono::DateTime<chrono::Utc>,
        offset: u64,
    ) -> Result<Offset, TriggerError>;

    /// Attach a subscriber to the live tail.
    ///
    /// `from_offset`, when set, is an **engine `_offset` lower bound**, not a
    /// driver-native sequence. The broker MUST begin delivery at or before
    /// that engine offset — delivering earlier events is permitted — so the
    /// caller is guaranteed never to miss an engine offset `>= from_offset`.
    /// It is the engine's subscribe seam ([`crate::trigger::Subscriber`]) that
    /// dedups the overlap by engine `_offset`; the broker is not required to
    /// start *exactly* at `from_offset`.
    ///
    /// This contract is what keeps the at-least-once guarantee correct across
    /// drivers whose native sequence is an independent counter from the engine
    /// `_offset` (JetStream's stream sequence): the engine never hands an
    /// engine offset to a driver as if it were a native sequence, because the
    /// two skew permanently after any post-commit fan-out failure (the
    /// best-effort path in [`crate::trigger::Publisher`]). A driver that cannot
    /// translate an engine offset into its own sequence MUST over-deliver
    /// (start from the earliest retained event) rather than guess a sequence.
    ///
    /// If `from_offset.is_some()` and the offset is older than what the driver
    /// retains, the broker returns [`TriggerError::OffsetEvicted`] — the
    /// engine's subscribe path falls back to backing-table replay for the
    /// missing prefix.
    async fn subscribe(
        &self,
        topic_id: TopicId,
        predicate: Predicate,
        from_offset: Option<Offset>,
    ) -> Result<Subscription, TriggerError>;

    /// Snapshot every consumer currently bound to `topic_id`. Returns one
    /// [`ConsumerOffsetSnapshot`] per consumer with the broker's
    /// last-delivered and ack-floor stream sequences. Used by the
    /// backup-restore path to capture consumer state so a fresh broker
    /// can be primed with the same offsets after a restore (a downstream
    /// consumer's backup module is the consuming side).
    ///
    /// Returns [`TriggerError::TopicNotFound`] when `topic_id` was never
    /// registered with this broker.
    async fn list_consumers(
        &self,
        topic_id: TopicId,
    ) -> Result<Vec<ConsumerOffsetSnapshot>, TriggerError>;

    /// Driver identity for telemetry and routing.
    fn driver_kind(&self) -> BrokerKind;
}

/// Discriminates the available broker implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrokerKind {
    InMemory,
    JetStream,
}
