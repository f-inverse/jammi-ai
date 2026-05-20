//! Live JetStream integration tests for the trigger-stream primitive.
//!
//! Gated behind `live-broker-tests`. The dedicated `test-broker` CI job
//! spins up a `nats:2.10` service container with `-js` and points the
//! tests at it via `JAMMI_TEST_NATS_URL`. Each test stands up its own
//! topic (UUIDv7 keeps names unique), exercises the same shape of
//! register-publish-subscribe-filter as the in-memory cases, then drops
//! the stream so the NATS server stays clean across runs.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use jammi_engine::trigger::{
    JetStreamBroker, Offset, Predicate, TopicDefinition, TopicId, TriggerBroker, TriggerError,
};

const ENV_VAR: &str = "JAMMI_TEST_NATS_URL";

async fn open_broker() -> JetStreamBroker {
    let url = std::env::var(ENV_VAR).unwrap_or_else(|_| {
        panic!("{ENV_VAR} is required for live-broker tests (e.g. nats://localhost:4222)")
    });
    JetStreamBroker::connect(&url, 60)
        .await
        .expect("connect to JetStream")
}

fn topic_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("value", DataType::Float64, true),
    ]))
}

fn make_topic(name: &str) -> TopicDefinition {
    TopicDefinition {
        id: TopicId::new(),
        name: name.into(),
        schema: topic_schema(),
        tenant: None,
        broker_metadata: BTreeMap::new(),
    }
}

fn batch_of(ids: &[i64], kinds: &[&str], values: &[f64]) -> RecordBatch {
    RecordBatch::try_new(
        topic_schema(),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(kinds.to_vec())),
            Arc::new(Float64Array::from(values.to_vec())),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn register_and_drop_round_trip() {
    let broker = open_broker().await;
    let topic = make_topic("live.smoke");
    broker.register_topic(&topic).await.unwrap();
    broker.drop_topic(topic.id).await.unwrap();
}

#[tokio::test]
async fn publish_subscribe_filter() {
    let broker = open_broker().await;
    let topic = make_topic("live.pubsub");
    broker.register_topic(&topic).await.unwrap();

    let session = datafusion::execution::context::SessionContext::new();
    let predicate = Predicate::from_sql(&session, Arc::clone(&topic.schema), "kind = 'X'").unwrap();
    let mut stream = broker
        .subscribe(
            topic.id,
            predicate,
            Some(Offset::new(0, chrono::Utc::now())),
        )
        .await
        .unwrap();

    // Publish even-i = 'X' / odd-i = 'Y' batches (single row each).
    for i in 0..20i64 {
        let kind = if i % 2 == 0 { "X" } else { "Y" };
        let batch = batch_of(&[i], &[kind], &[i as f64]);
        broker
            .publish(topic.id, batch, chrono::Utc::now(), i as u64)
            .await
            .unwrap();
    }

    let mut matched = 0;
    while matched < 10 {
        let delivered = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("subscribe stream timed out")
            .expect("stream ended early")
            .unwrap();
        let kinds = delivered
            .batch
            .column_by_name("kind")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..kinds.len() {
            assert_eq!(kinds.value(i), "X");
        }
        matched += 1;
    }
    drop(stream);
    broker.drop_topic(topic.id).await.unwrap();
}

#[tokio::test]
async fn publish_to_unregistered_topic_is_not_found() {
    let broker = open_broker().await;
    let stray = TopicId::new();
    let batch = batch_of(&[1], &["X"], &[1.0]);
    match broker.publish(stray, batch, chrono::Utc::now(), 0).await {
        Err(TriggerError::TopicNotFound(_)) => {}
        other => panic!("expected TopicNotFound, got {other:?}"),
    }
}
