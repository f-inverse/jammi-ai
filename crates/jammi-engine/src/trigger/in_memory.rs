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
pub struct InMemoryBroker {
    channels: RwLock<HashMap<TopicId, ChannelState>>,
    capacity: usize,
}

impl InMemoryBroker {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CHANNEL_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            channels: RwLock::new(HashMap::new()),
            capacity,
        }
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
