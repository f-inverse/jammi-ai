# Subscribe to a Topic with a SQL Predicate Filter

A subscription tails a topic and yields only the batches whose rows
satisfy a SQL `WHERE`-clause predicate. The predicate is parsed once
through DataFusion at subscribe time; the broker delivers each batch,
the engine evaluates the predicate against it, and rows that match
flow to the consumer.

Reach for predicate-filtered subscriptions when different downstream
consumers need different selectivity on the same event stream — a
search-index that only cares about `op = 'c'`, an audit log that
wants every event, a cache invalidator that wants `op IN ('u', 'd')`.

## Goal

Open a subscription on the `cdc.orders` topic with a predicate that
matches only deletes, and consume the stream from Rust.

## Setup

Assumes the topic was registered (see
[Publish Events to a Topic](./publish-events.md)) and you have a
`Subscriber` constructed against the same broker the publisher uses.

## Open the subscription

```rust,no_run
# extern crate jammi_db;
# extern crate arrow;
# extern crate arrow_schema;
# extern crate datafusion;
# extern crate futures;
# extern crate tokio;
# use std::sync::Arc;
# use arrow_schema::SchemaRef;
# use datafusion::execution::context::SessionContext;
# use futures::StreamExt;
# use jammi_db::trigger::{Predicate, Subscriber, TopicDefinition};
# async fn ex(
#     subscriber: &Subscriber,
#     session: &SessionContext,
#     topic: &TopicDefinition,
# ) -> Result<(), jammi_db::trigger::TriggerError> {
let predicate = Predicate::from_sql(session, Arc::clone(&topic.schema), "op = 'd'")?;

let mut stream = subscriber
    .subscribe(topic, predicate, None /* from_offset: None = live tail */)
    .await?;

while let Some(delivered) = stream.next().await {
    let batch = delivered?;
    handle_deletes(batch.batch);
}
# Ok(())
# }
# fn handle_deletes(_: arrow::array::RecordBatch) {}
```

`from_offset = None` starts the subscription at the broker's live tail
(no replay). `Some(0)` starts from the earliest retained event; the
engine joins backing-table replay with the live broker stream so the
client sees one continuous sequence of `DeliveredBatch`.

## Predicate dialect

Predicates are a *subset* of DataFusion SQL. The whitelist:

| Supported                                | Rejected                          |
|------------------------------------------|-----------------------------------|
| Column references (`col`)                | Subqueries (`SELECT …`)           |
| Literal scalars (`1`, `'foo'`, `true`)   | Aggregates (`SUM`, `COUNT`, …)    |
| Comparison ops (`=`, `<`, `>`, `<=`, `>=`, `!=`) | Window functions          |
| Boolean ops (`AND`, `OR`, `NOT`)         | Joins                             |
| `IS NULL`, `IS NOT NULL`                 | `CASE WHEN`                       |
| `IN (literal, literal, …)`               | Functions outside the whitelist   |
| `LIKE`, `BETWEEN`                        |                                   |
| Whitelisted string functions             |                                   |

The string-function whitelist is `lower`, `upper`, `length`,
`starts_with`, `ends_with`. Anything outside this list returns
`PredicateUnsupported` at subscribe time — the stream never opens.
An unparseable predicate returns `PredicateParse` for the same
reason.

## Reconnection and replay

If your consumer disconnects and reconnects, pass the last-seen
offset as `from_offset` to resume without missing events:

```rust,no_run
# extern crate jammi_db;
# extern crate chrono;
# extern crate tokio;
# use chrono::Utc;
# use std::sync::Arc;
# use jammi_db::trigger::{Offset, Predicate, Subscriber, TopicDefinition};
# async fn ex(
#     subscriber: &Subscriber,
#     topic: &TopicDefinition,
#     last_seen: u64,
# ) -> Result<(), jammi_db::trigger::TriggerError> {
let resume_from = Offset::new(last_seen + 1, Utc::now());
let _stream = subscriber
    .subscribe(topic, Predicate::match_all(), Some(resume_from))
    .await?;
# Ok(())
# }
```

The engine reads the backing table for offsets `>= resume_from`, then
attaches the broker live stream starting strictly above the last
replayed offset — the two halves never deliver the same offset twice.

## Backpressure

A slow consumer slows the producer; events are *not* dropped. The
chain is: the broker tail backs up onto a bounded `mpsc::channel`, the
channel's `send()` future awaits, the broker poll loop pauses,
publishers awaiting the broker fan-out experience matching back-
pressure. The backing table — the authoritative log — is still
written without delay, so a consumer that disconnects under load can
always catch up via replay.

## See also

- [Publish Events to a Topic](./publish-events.md) — the publisher side.
- [Replay Events from the Backing Table](./replay-from-backing-table.md)
  — bypass the broker entirely and read the event log via Flight SQL.
