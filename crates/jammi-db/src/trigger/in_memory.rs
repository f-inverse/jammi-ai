//! In-memory `TriggerBroker` backed by `tokio::sync::broadcast`.
//!
//! Default broker for unit tests and single-process deployments. Lagged
//! receivers are surfaced as [`TriggerError::OffsetEvicted`] so the
//! engine's subscribe path can route the missing prefix through
//! backing-table replay.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use arrow_schema::SchemaRef;
use async_stream::try_stream;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use tokio::sync::broadcast;

use crate::trigger::broker::{BrokerKind, TriggerBroker};
use crate::trigger::error::TriggerError;
use crate::trigger::ids::{SubscriptionId, TopicId};
use crate::trigger::offset::Offset;
use crate::trigger::predicate::Predicate;
use crate::trigger::subscription::{DeliveredBatch, Subscription};
use crate::trigger::topic::TopicDefinition;

/// Per-topic broadcast channel and its committed schema (for the idempotency
/// check during re-registration).
struct ChannelState {
    sender: broadcast::Sender<DeliveredBatch>,
    schema: SchemaRef,
}

/// Capacity of each topic's broadcast channel. Lagged receivers fall back to
/// backing-table replay (see `Subscriber::subscribe`).
pub const DEFAULT_CHANNEL_CAPACITY: usize = 1024;

/// `TriggerBroker` implementation that holds every channel in process memory.
///
/// Holds an optional pending-failure slot — when set, the next `publish` call
/// consumes it and returns the corresponding [`TriggerError::Driver`] instead
/// of broadcasting. Used by tests that need to exercise publisher failure
/// paths (e.g. the enterprise gate's `publish_failure_does_not_fail_check`
/// invariant) deterministically.
pub struct InMemoryBroker {
    channels: RwLock<HashMap<TopicId, ChannelState>>,
    capacity: usize,
    pending_failure: RwLock<Option<String>>,
}

impl InMemoryBroker {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CHANNEL_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            channels: RwLock::new(HashMap::new()),
            capacity,
            pending_failure: RwLock::new(None),
        }
    }

    /// Arm the broker so the next `publish` call returns
    /// [`TriggerError::Driver`] with `message` and clears the arm. Subsequent
    /// publishes (after the armed one) succeed normally. Useful for tests
    /// that verify publisher-failure handling without depending on a real
    /// broker outage.
    pub fn trigger_failure_for_next_publish(&self, message: impl Into<String>) {
        *self.pending_failure.write() = Some(message.into());
    }
}

impl Default for InMemoryBroker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TriggerBroker for InMemoryBroker {
    async fn register_topic(&self, topic: &TopicDefinition) -> Result<(), TriggerError> {
        let mut channels = self.channels.write();
        if let Some(existing) = channels.get(&topic.id) {
            if existing.schema.as_ref() != topic.schema.as_ref() {
                return Err(TriggerError::SchemaConflict {
                    topic: topic.name.clone(),
                    detail: "in-memory broker already holds a different schema for this topic"
                        .into(),
                });
            }
            return Ok(());
        }
        let (sender, _initial_rx) = broadcast::channel(self.capacity);
        channels.insert(
            topic.id,
            ChannelState {
                sender,
                schema: Arc::clone(&topic.schema),
            },
        );
        Ok(())
    }

    async fn drop_topic(&self, topic_id: TopicId) -> Result<(), TriggerError> {
        self.channels.write().remove(&topic_id);
        Ok(())
    }

    async fn publish(
        &self,
        topic_id: TopicId,
        batch: RecordBatch,
        produced_at: DateTime<Utc>,
        offset: u64,
    ) -> Result<Offset, TriggerError> {
        if let Some(message) = self.pending_failure.write().take() {
            return Err(TriggerError::Driver(message));
        }
        let off = Offset::new(offset, produced_at);
        let sender = self
            .channels
            .read()
            .get(&topic_id)
            .map(|s| s.sender.clone())
            .ok_or_else(|| TriggerError::TopicNotFound(topic_id.to_string()))?;
        // `Sender::send` returns Err only when there are zero receivers,
        // which is not a delivery failure for fan-out semantics.
        let _ = sender.send(DeliveredBatch {
            offset: off,
            produced_at,
            batch,
        });
        Ok(off)
    }

    async fn subscribe(
        &self,
        topic_id: TopicId,
        predicate: Predicate,
        from_offset: Option<Offset>,
    ) -> Result<Subscription, TriggerError> {
        let mut rx = self
            .channels
            .read()
            .get(&topic_id)
            .map(|s| s.sender.subscribe())
            .ok_or_else(|| TriggerError::TopicNotFound(topic_id.to_string()))?;
        let after = from_offset.map(|o| o.value());
        let stream = try_stream! {
            loop {
                match rx.recv().await {
                    Ok(d) => {
                        if let Some(threshold) = after {
                            if d.offset.value() < threshold {
                                continue;
                            }
                        }
                        if let Some(filtered) = predicate.evaluate(&d.batch)? {
                            yield DeliveredBatch {
                                offset: d.offset,
                                produced_at: d.produced_at,
                                batch: filtered,
                            };
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        Err(TriggerError::OffsetEvicted(n))?;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        };
        Ok(Subscription::new(SubscriptionId::new(), Box::pin(stream)))
    }

    fn driver_kind(&self) -> BrokerKind {
        BrokerKind::InMemory
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow_schema::{DataType, Field, Schema};
    use std::collections::BTreeMap;

    fn topic(name: &str) -> TopicDefinition {
        TopicDefinition {
            id: TopicId::new(),
            name: name.into(),
            schema: Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)])),
            tenant: None,
            broker_metadata: BTreeMap::new(),
        }
    }

    fn batch(schema: &SchemaRef, values: &[i64]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::clone(schema),
            vec![Arc::new(Int64Array::from(values.to_vec()))],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn trigger_failure_consumes_one_publish() {
        let broker = InMemoryBroker::new();
        let t = topic("test.failure");
        broker.register_topic(&t).await.unwrap();

        broker.trigger_failure_for_next_publish("simulated broker outage");
        let err = broker
            .publish(t.id, batch(&t.schema, &[1]), Utc::now(), 0)
            .await
            .unwrap_err();
        match err {
            TriggerError::Driver(msg) => assert_eq!(msg, "simulated broker outage"),
            other => panic!("expected Driver error, got {other:?}"),
        }

        // The next publish must succeed — the failure was one-shot.
        broker
            .publish(t.id, batch(&t.schema, &[2]), Utc::now(), 1)
            .await
            .expect("subsequent publish succeeds");
    }

    #[tokio::test]
    async fn trigger_failure_unarmed_publish_succeeds() {
        let broker = InMemoryBroker::new();
        let t = topic("test.unarmed");
        broker.register_topic(&t).await.unwrap();
        broker
            .publish(t.id, batch(&t.schema, &[42]), Utc::now(), 0)
            .await
            .expect("publish without arm succeeds");
    }
}
