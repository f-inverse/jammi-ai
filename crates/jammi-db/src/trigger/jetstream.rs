//! NATS JetStream `TriggerBroker` implementation.
//!
//! Mounted behind the `jetstream-broker` cargo feature so default builds do
//! not pull `async-nats` into the dependency tree. Maps each topic to a
//! JetStream stream named `jammi_topic_<topic_id>` with one subject
//! `jammi.topic.<id>.batch`; publishes carry the Arrow IPC payload as the
//! NATS message body and the engine-assigned offset / produced-at instant
//! as message headers. Subscribes use an ordered pull consumer with a
//! `DeliverByStartSequence` policy translated from `from_offset`.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use arrow::record_batch::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::SchemaRef;
use async_nats::header::HeaderMap;
use async_nats::jetstream::consumer::{self, DeliverPolicy};
use async_nats::jetstream::stream::{Config as StreamConfig, RetentionPolicy};
use async_nats::jetstream::{self, Context as JetStreamContext};
use async_stream::try_stream;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use parking_lot::RwLock;
use std::collections::HashMap;

use crate::trigger::broker::{BrokerKind, TriggerBroker};
use crate::trigger::error::TriggerError;
use crate::trigger::ids::{SubscriptionId, TopicId};
use crate::trigger::offset::Offset;
use crate::trigger::predicate::Predicate;
use crate::trigger::subscription::{DeliveredBatch, Subscription};
use crate::trigger::topic::TopicDefinition;

/// Headers attached to every published JetStream message. Receivers read
/// them to reconstruct the engine-assigned `Offset` without paying for
/// schema decoding twice.
const HDR_OFFSET: &str = "jammi-offset";
const HDR_PRODUCED_AT: &str = "jammi-produced-at-us";

/// Production broker driver backed by NATS JetStream. Holds the
/// `async-nats` JetStream context plus a small cache of per-topic Arrow
/// schemas — published messages carry the IPC schema header and a batch
/// message, but subscribers benefit from a single decoded schema kept
/// alongside the topic registration.
pub struct JetStreamBroker {
    context: JetStreamContext,
    schemas: RwLock<HashMap<TopicId, SchemaRef>>,
    retention: Duration,
}

impl JetStreamBroker {
    /// Open an anonymous broker bound to the NATS server at `url`.
    /// `retention_seconds` applies to every freshly-created stream —
    /// operator overrides via `TopicDefinition.broker_metadata` are honoured
    /// by `register_topic`.
    pub async fn connect(url: &str, retention_seconds: u64) -> Result<Self, TriggerError> {
        let client = async_nats::connect(url)
            .await
            .map_err(|e| TriggerError::Driver(format!("nats connect: {e}")))?;
        Ok(Self::from_client(client, retention_seconds))
    }

    /// Open a broker authenticated with a NATS `.creds` file. Use this
    /// constructor for SaaS deployments where the broker rejects anonymous
    /// connections.
    pub async fn connect_with_credentials(
        url: &str,
        retention_seconds: u64,
        credentials_path: &Path,
    ) -> Result<Self, TriggerError> {
        let client = async_nats::ConnectOptions::with_credentials_file(credentials_path)
            .await
            .map_err(|e| TriggerError::Driver(format!("nats creds: {e}")))?
            .connect(url)
            .await
            .map_err(|e| TriggerError::Driver(format!("nats connect: {e}")))?;
        Ok(Self::from_client(client, retention_seconds))
    }

    /// Wrap a connected `async_nats::Client` into a [`JetStreamBroker`]
    /// with the given retention. Shared by [`Self::connect`] and
    /// [`Self::connect_with_credentials`] so the two paths agree on the
    /// schemas-cache shape and the `JetStreamContext` derivation.
    fn from_client(client: async_nats::Client, retention_seconds: u64) -> Self {
        Self {
            context: jetstream::new(client),
            schemas: RwLock::new(HashMap::new()),
            retention: Duration::from_secs(retention_seconds),
        }
    }

    fn stream_name(topic_id: TopicId) -> String {
        format!("jammi_topic_{}", topic_id.as_uuid().simple())
    }

    fn subject_for(topic_id: TopicId) -> String {
        format!("jammi.topic.{}.batch", topic_id.as_uuid().simple())
    }

    fn retention_for(&self, topic: &TopicDefinition) -> Duration {
        topic
            .broker_metadata
            .get("retention_seconds")
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(self.retention)
    }
}

#[async_trait]
impl TriggerBroker for JetStreamBroker {
    async fn register_topic(&self, topic: &TopicDefinition) -> Result<(), TriggerError> {
        let cfg = StreamConfig {
            name: Self::stream_name(topic.id),
            subjects: vec![Self::subject_for(topic.id)],
            retention: RetentionPolicy::Limits,
            max_age: self.retention_for(topic),
            ..Default::default()
        };
        self.context
            .get_or_create_stream(cfg)
            .await
            .map_err(|e| TriggerError::Driver(format!("create_stream: {e}")))?;
        self.schemas
            .write()
            .insert(topic.id, Arc::clone(&topic.schema));
        Ok(())
    }

    async fn drop_topic(&self, topic_id: TopicId) -> Result<(), TriggerError> {
        let name = Self::stream_name(topic_id);
        self.context
            .delete_stream(&name)
            .await
            .map_err(|e| TriggerError::Driver(format!("delete_stream {name}: {e}")))?;
        self.schemas.write().remove(&topic_id);
        Ok(())
    }

    async fn publish(
        &self,
        topic_id: TopicId,
        batch: RecordBatch,
        produced_at: DateTime<Utc>,
        offset: u64,
    ) -> Result<Offset, TriggerError> {
        let schema = self
            .schemas
            .read()
            .get(&topic_id)
            .cloned()
            .ok_or_else(|| TriggerError::TopicNotFound(topic_id.to_string()))?;
        let payload = encode_batch_ipc(&schema, &batch)?;

        let mut headers = HeaderMap::new();
        headers.insert(HDR_OFFSET, offset.to_string());
        headers.insert(HDR_PRODUCED_AT, produced_at.timestamp_micros().to_string());

        let subject = Self::subject_for(topic_id);
        self.context
            .publish_with_headers(subject, headers, payload.into())
            .await
            .map_err(|e| TriggerError::Driver(format!("publish: {e}")))?
            .await
            .map_err(|e| TriggerError::Driver(format!("publish-ack: {e}")))?;
        Ok(Offset::new(offset, produced_at))
    }

    async fn subscribe(
        &self,
        topic_id: TopicId,
        predicate: Predicate,
        from_offset: Option<Offset>,
    ) -> Result<Subscription, TriggerError> {
        let schema = self
            .schemas
            .read()
            .get(&topic_id)
            .cloned()
            .ok_or_else(|| TriggerError::TopicNotFound(topic_id.to_string()))?;
        let stream_name = Self::stream_name(topic_id);
        let stream = self
            .context
            .get_stream(&stream_name)
            .await
            .map_err(|e| TriggerError::Driver(format!("get_stream {stream_name}: {e}")))?;

        let deliver_policy = match from_offset {
            None => DeliverPolicy::New,
            // `Offset(0)` means "from the earliest retained batch"; JetStream's
            // `DeliverAll` covers that without requiring the start sequence to
            // exist in the retained window.
            Some(off) if off.value() == 0 => DeliverPolicy::All,
            Some(off) => DeliverPolicy::ByStartSequence {
                start_sequence: off.value(),
            },
        };

        let consumer = stream
            .create_consumer(consumer::pull::Config {
                deliver_policy,
                filter_subject: Self::subject_for(topic_id),
                ack_policy: consumer::AckPolicy::Explicit,
                ..Default::default()
            })
            .await
            .map_err(|e| TriggerError::Driver(format!("create_consumer: {e}")))?;

        let mut messages = consumer
            .messages()
            .await
            .map_err(|e| TriggerError::Driver(format!("consumer messages: {e}")))?;

        let inner = try_stream! {
            while let Some(message) = messages.next().await {
                let message = message
                    .map_err(|e| TriggerError::Driver(format!("recv: {e}")))?;
                let delivered = decode_message(&schema, &message)?;
                message
                    .ack()
                    .await
                    .map_err(|e| TriggerError::Driver(format!("ack: {e}")))?;
                if let Some(filtered) = predicate.evaluate(&delivered.batch)? {
                    yield DeliveredBatch {
                        offset: delivered.offset,
                        produced_at: delivered.produced_at,
                        batch: filtered,
                    };
                }
            }
        };
        Ok(Subscription::new(SubscriptionId::new(), Box::pin(inner)))
    }

    fn driver_kind(&self) -> BrokerKind {
        BrokerKind::JetStream
    }
}

fn encode_batch_ipc(schema: &SchemaRef, batch: &RecordBatch) -> Result<Vec<u8>, TriggerError> {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, schema.as_ref())
            .map_err(|e| TriggerError::Driver(format!("ipc encode: {e}")))?;
        writer
            .write(batch)
            .map_err(|e| TriggerError::Driver(format!("ipc encode batch: {e}")))?;
        writer
            .finish()
            .map_err(|e| TriggerError::Driver(format!("ipc finish: {e}")))?;
    }
    Ok(buf)
}

fn decode_message(
    schema: &SchemaRef,
    message: &async_nats::jetstream::Message,
) -> Result<DeliveredBatch, TriggerError> {
    let headers = message
        .headers
        .as_ref()
        .ok_or_else(|| TriggerError::Driver("message missing headers".into()))?;
    let offset_value: u64 = headers
        .get(HDR_OFFSET)
        .ok_or_else(|| TriggerError::Driver(format!("message missing `{HDR_OFFSET}`")))?
        .as_str()
        .parse()
        .map_err(|e| TriggerError::Driver(format!("`{HDR_OFFSET}` parse: {e}")))?;
    let produced_at_us: i64 = headers
        .get(HDR_PRODUCED_AT)
        .ok_or_else(|| TriggerError::Driver(format!("message missing `{HDR_PRODUCED_AT}`")))?
        .as_str()
        .parse()
        .map_err(|e| TriggerError::Driver(format!("`{HDR_PRODUCED_AT}` parse: {e}")))?;
    let produced_at = DateTime::<Utc>::from_timestamp_micros(produced_at_us).ok_or_else(|| {
        TriggerError::Driver(format!(
            "`{HDR_PRODUCED_AT}` out of range: {produced_at_us}"
        ))
    })?;
    let cursor = std::io::Cursor::new(&message.payload[..]);
    let mut reader = StreamReader::try_new(cursor, None)
        .map_err(|e| TriggerError::Driver(format!("ipc decode: {e}")))?;
    let batch = reader
        .next()
        .ok_or_else(|| TriggerError::Driver("ipc stream had no batch".into()))?
        .map_err(|e| TriggerError::Driver(format!("ipc decode batch: {e}")))?;
    if batch.schema().as_ref() != schema.as_ref() {
        return Err(TriggerError::Driver(
            "message schema does not match topic schema".into(),
        ));
    }
    Ok(DeliveredBatch {
        offset: Offset::new(offset_value, produced_at),
        produced_at,
        batch,
    })
}
