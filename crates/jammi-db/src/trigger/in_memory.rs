//! In-memory `TriggerBroker` backed by `tokio::sync::broadcast`.
//!
//! Default broker for unit tests and single-process deployments. Lagged
//! receivers are surfaced as [`TriggerError::OffsetEvicted`] so the
//! engine's subscribe path can route the missing prefix through
//! backing-table replay.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use arrow::record_batch::RecordBatch;
use arrow_schema::SchemaRef;
use async_stream::try_stream;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use tokio::sync::broadcast;

use crate::trigger::broker::{BrokerKind, TriggerBroker};
use crate::trigger::consumer::ConsumerOffsetSnapshot;
use crate::trigger::error::TriggerError;
use crate::trigger::ids::{SubscriptionId, TopicId};
use crate::trigger::offset::Offset;
use crate::trigger::predicate::Predicate;
use crate::trigger::subscription::{DeliveredBatch, Subscription};
use crate::trigger::topic::TopicDefinition;

/// Per-subscription bookkeeping the broker hands back through
/// [`TriggerBroker::list_consumers`]. The subscription's async stream
/// owns the `Arc`; the channel state holds a `Weak` so dropping the
/// stream automatically prunes the consumer from the listing.
struct ConsumerTracker {
    consumer_name: String,
    topic_id: TopicId,
    last_delivered: AtomicU64,
}

/// Per-topic broadcast channel, its committed schema (for the idempotency
/// check during re-registration), and weak references to every live
/// subscription on the topic.
struct ChannelState {
    sender: broadcast::Sender<DeliveredBatch>,
    schema: SchemaRef,
    consumers: Vec<Weak<ConsumerTracker>>,
}

/// Capacity of each topic's broadcast channel. Lagged receivers fall back to
/// backing-table replay (see `Subscriber::subscribe`).
pub const DEFAULT_CHANNEL_CAPACITY: usize = 1024;

/// `TriggerBroker` implementation that holds every channel in process memory.
///
/// Holds an optional pending-failure slot — when set, the next `publish` call
/// consumes it and returns the corresponding [`TriggerError::Driver`] instead
/// of broadcasting. Used by tests that need to exercise publisher failure
/// paths (e.g. a downstream consumer's `publish_failure_does_not_fail_check`
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
                consumers: Vec::new(),
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
        let subscription_id = SubscriptionId::new();
        let tracker = Arc::new(ConsumerTracker {
            consumer_name: subscription_id.to_string(),
            topic_id,
            last_delivered: AtomicU64::new(0),
        });

        let mut rx = {
            let mut channels = self.channels.write();
            let state = channels
                .get_mut(&topic_id)
                .ok_or_else(|| TriggerError::TopicNotFound(topic_id.to_string()))?;
            // Drop tracker weaks that have already been collected so the
            // per-topic vector does not grow without bound across the
            // lifetime of long-lived brokers (UAT, dashboards).
            state.consumers.retain(|w| w.strong_count() > 0);
            state.consumers.push(Arc::downgrade(&tracker));
            state.sender.subscribe()
        };
        let after = from_offset.map(|o| o.value());
        // The stream takes ownership of `tracker` so it stays alive for the
        // lifetime of the subscription. When the consumer drops the
        // `Subscription`, the tracker is dropped and the broker's `Weak`
        // entry stops upgrading — `list_consumers` filters it out.
        let stream = try_stream! {
            let _tracker_guard = Arc::clone(&tracker);
            loop {
                match rx.recv().await {
                    Ok(d) => {
                        if let Some(threshold) = after {
                            if d.offset.value() < threshold {
                                continue;
                            }
                        }
                        tracker
                            .last_delivered
                            .store(d.offset.value(), Ordering::Relaxed);
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
        Ok(Subscription::new(subscription_id, Box::pin(stream)))
    }

    async fn list_consumers(
        &self,
        topic_id: TopicId,
    ) -> Result<Vec<ConsumerOffsetSnapshot>, TriggerError> {
        let mut channels = self.channels.write();
        let state = channels
            .get_mut(&topic_id)
            .ok_or_else(|| TriggerError::TopicNotFound(topic_id.to_string()))?;
        let mut snapshots: Vec<ConsumerOffsetSnapshot> = Vec::with_capacity(state.consumers.len());
        state.consumers.retain(|w| {
            if let Some(tracker) = w.upgrade() {
                let last_delivered = tracker.last_delivered.load(Ordering::Relaxed);
                snapshots.push(ConsumerOffsetSnapshot {
                    consumer_name: tracker.consumer_name.clone(),
                    topic_id: tracker.topic_id,
                    last_delivered_stream_sequence: last_delivered,
                    // The in-memory broker delivers via `tokio::broadcast`,
                    // which has no explicit-ack model; every received batch
                    // is implicitly acknowledged the moment the subscriber
                    // observes it. Surfacing the same value for both fields
                    // keeps the round-trip stable through the backup-restore
                    // path (`last_ack` is the field a restore would use to
                    // resume).
                    last_ack_stream_sequence: last_delivered,
                });
                true
            } else {
                false
            }
        });
        Ok(snapshots)
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

    #[tokio::test]
    async fn list_consumers_on_unknown_topic_returns_not_found() {
        let broker = InMemoryBroker::new();
        match broker.list_consumers(TopicId::new()).await {
            Err(TriggerError::TopicNotFound(_)) => {}
            other => panic!("expected TopicNotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn list_consumers_on_topic_without_subscribers_is_empty() {
        let broker = InMemoryBroker::new();
        let t = topic("test.no_subscribers");
        broker.register_topic(&t).await.unwrap();
        let consumers = broker.list_consumers(t.id).await.unwrap();
        assert!(consumers.is_empty());
    }

    #[tokio::test]
    async fn list_consumers_prunes_dropped_subscriptions() {
        let broker = InMemoryBroker::new();
        let t = topic("test.drop_prunes");
        broker.register_topic(&t).await.unwrap();
        {
            let _sub = broker
                .subscribe(t.id, Predicate::match_all(), None)
                .await
                .unwrap();
            let alive = broker.list_consumers(t.id).await.unwrap();
            assert_eq!(alive.len(), 1, "subscription must register a consumer");
        }
        // After the subscription scope ends the tracker's strong count drops
        // to zero; `list_consumers` upgrades the weaks and prunes the dead
        // entry, returning an empty list.
        let after_drop = broker.list_consumers(t.id).await.unwrap();
        assert!(
            after_drop.is_empty(),
            "dropped subscription must not appear in list_consumers"
        );
    }
}
