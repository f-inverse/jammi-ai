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
use jammi_db::trigger::{
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
async fn list_consumers_returns_jetstream_consumer_info() {
    // Mirrors the in-memory `list_consumers` integration test against a real
    // JetStream server. Subscribe twice, publish, drain, then confirm both
    // consumer names + the broker's `delivered.stream_sequence` round-trip
    // through `ConsumerOffsetSnapshot`.
    let broker = open_broker().await;
    let topic = make_topic("live.list_consumers");
    broker.register_topic(&topic).await.unwrap();

    let mut sub_a = broker
        .subscribe(
            topic.id,
            Predicate::match_all(),
            Some(Offset::new(0, chrono::Utc::now())),
        )
        .await
        .unwrap();
    let mut sub_b = broker
        .subscribe(
            topic.id,
            Predicate::match_all(),
            Some(Offset::new(0, chrono::Utc::now())),
        )
        .await
        .unwrap();

    for i in 0..3i64 {
        let batch = batch_of(&[i], &["X"], &[i as f64]);
        broker
            .publish(topic.id, batch, chrono::Utc::now(), i as u64)
            .await
            .unwrap();
    }

    // Drain three batches per subscriber so JetStream advances each
    // consumer's ack floor to the head of the stream before we list.
    for _ in 0..3 {
        let _ = tokio::time::timeout(Duration::from_secs(5), sub_a.next())
            .await
            .expect("sub_a timed out")
            .expect("sub_a stream ended early")
            .unwrap();
        let _ = tokio::time::timeout(Duration::from_secs(5), sub_b.next())
            .await
            .expect("sub_b timed out")
            .expect("sub_b stream ended early")
            .unwrap();
    }

    let snapshots = broker.list_consumers(topic.id).await.unwrap();
    assert_eq!(
        snapshots.len(),
        2,
        "expected one snapshot per live JetStream consumer, got {snapshots:?}"
    );
    for snap in &snapshots {
        assert_eq!(snap.topic_id, topic.id);
        // The first published offset is 0, so three batches advance the
        // stream sequence to 3 (JetStream sequences are 1-based per
        // message).
        assert_eq!(
            snap.last_delivered_stream_sequence, 3,
            "consumer {} should be at stream sequence 3 after draining three batches",
            snap.consumer_name
        );
        assert_eq!(
            snap.last_ack_stream_sequence, snap.last_delivered_stream_sequence,
            "every delivered batch was acked in the drain loop above"
        );
    }

    drop(sub_a);
    drop(sub_b);
    broker.drop_topic(topic.id).await.unwrap();
}

#[tokio::test]
async fn list_consumers_on_unknown_topic_is_not_found() {
    let broker = open_broker().await;
    match broker.list_consumers(TopicId::new()).await {
        Err(TriggerError::TopicNotFound(_)) => {}
        other => panic!("expected TopicNotFound, got {other:?}"),
    }
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

#[tokio::test]
async fn consumer_recreate_resumes_engine_offsets_with_no_loss() {
    // Track T1 — JetStream consumer-recreate + resume, the test that
    // demonstrates the replay/live seam bug and its fix on a real broker.
    //
    // The engine `_offset` is ABSOLUTE: it is seeded from `MAX(_offset)` on the
    // durable backing table, so after a restart (or for any topic whose early
    // events have aged out of the broker's retention window) the live publishes
    // begin at a high engine offset — here `BASE` — while a freshly-created
    // JetStream stream numbers its own messages from sequence 1. The engine
    // `_offset` and the JetStream stream sequence are therefore independent
    // counters with a large, permanent skew: engine offset `BASE + k` lives at
    // JetStream stream sequence `k + 1`.
    //
    // Consume K engine offsets, drop the subscription (consumer-recreate /
    // broker-restart resume), then recreate from `last_seen + 1` — in
    // ENGINE-OFFSET space, exactly what the engine subscribe seam passes.
    // Every engine offset `> last_seen` must resume with NO LOSS.
    //
    // Before the seam fix this FAILS HARD: `JetStreamBroker::subscribe` mapped
    // the engine offset onto `DeliverPolicy::ByStartSequence { start_sequence }`,
    // conflating the engine `_offset` (`~BASE`) with the stream sequence
    // (`~K`). `ByStartSequence { last_seen + 1 }` asks for a stream sequence far
    // beyond the stream's head, so the consumer receives NOTHING and the resume
    // loop times out — total loss of the resumed suffix. After the fix
    // `subscribe(Some(_))` over-delivers from the earliest retained event
    // (`DeliverAll`) and the consumer dedups by engine `_offset`, so the
    // remaining `[last_seen+1 .. BASE+N)` resumes with no loss.
    let broker = open_broker().await;
    let topic = make_topic("live.consumer_recreate_resume");
    broker.register_topic(&topic).await.unwrap();

    // Absolute engine-offset base, far above any stream sequence this short
    // stream will reach, so `ByStartSequence { engine_offset }` overshoots the
    // stream head entirely.
    const BASE: u64 = 1_000;
    const N: u64 = 12;
    for k in 0..N {
        let off = BASE + k;
        let batch = batch_of(&[off as i64], &["X"], &[off as f64]);
        broker
            .publish(topic.id, batch, chrono::Utc::now(), off)
            .await
            .unwrap();
    }

    // First subscription: replay from the earliest engine offset, consume K,
    // remembering the highest engine offset seen.
    const K: usize = 5;
    let mut last_seen: Option<u64> = None;
    {
        let mut sub = broker
            .subscribe(
                topic.id,
                Predicate::match_all(),
                Some(Offset::new(BASE, chrono::Utc::now())),
            )
            .await
            .unwrap();
        for _ in 0..K {
            let delivered = tokio::time::timeout(Duration::from_secs(5), sub.next())
                .await
                .expect("first subscription timed out")
                .expect("stream ended early")
                .unwrap();
            last_seen = Some(delivered.offset.value());
        }
        // Drop the subscription — simulates a consumer-recreate / restart.
    }
    let last_seen = last_seen.expect("consumed at least one batch");

    // Recreate from `last_seen + 1` (engine-offset space). Dedup by engine
    // `_offset` exactly as `Subscriber::subscribe_scoped` does, discarding the
    // over-delivered prefix `<= last_seen`.
    let resume_from = last_seen + 1;
    let mut sub = broker
        .subscribe(
            topic.id,
            Predicate::match_all(),
            Some(Offset::new(resume_from, chrono::Utc::now())),
        )
        .await
        .unwrap();

    let expected: Vec<u64> = (resume_from..BASE + N).collect();
    let mut resumed: Vec<u64> = Vec::new();
    while resumed.len() < expected.len() {
        let delivered = tokio::time::timeout(Duration::from_secs(5), sub.next())
            .await
            .expect(
                "resumed subscription timed out — the engine offset was conflated with the JetStream stream sequence and the live tail started past the stream head (seam bug)",
            )
            .expect("stream ended early")
            .unwrap();
        let off = delivered.offset.value();
        if off >= resume_from {
            resumed.push(off);
        }
    }

    drop(sub);
    broker.drop_topic(topic.id).await.unwrap();

    let mut deduped = resumed.clone();
    deduped.dedup();
    assert_eq!(
        deduped, expected,
        "every engine offset in [{resume_from}..{}) must resume with no loss despite the engine-offset/stream-sequence skew; saw {resumed:?}",
        BASE + N
    );
}
