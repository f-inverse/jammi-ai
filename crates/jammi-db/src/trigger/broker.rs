//! `TriggerBroker` trait — pluggable transport for trigger-stream topics.

use arrow::record_batch::RecordBatch;
use async_trait::async_trait;

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

    /// Attach a subscriber. If `from_offset.is_some()` and the offset is
    /// older than what the driver retains, the broker returns
    /// [`TriggerError::OffsetEvicted`] — the engine's subscribe path falls
    /// back to backing-table replay for the missing prefix.
    async fn subscribe(
        &self,
        topic_id: TopicId,
        predicate: Predicate,
        from_offset: Option<Offset>,
    ) -> Result<Subscription, TriggerError>;

    /// Driver identity for telemetry and routing.
    fn driver_kind(&self) -> BrokerKind;
}

/// Discriminates the available broker implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrokerKind {
    InMemory,
    JetStream,
}
