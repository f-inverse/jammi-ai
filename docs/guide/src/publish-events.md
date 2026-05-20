# Publish Events to a Topic

A *trigger-stream topic* is a catalog-registered Arrow schema plus a
backing mutable table. Publishers append `RecordBatch`es; subscribers
filter and receive them. The engine owns the offset counter and the
durable event log; the broker (in-memory by default, NATS JetStream
in clustered deployments) fans live deliveries out to attached
subscribers.

Reach for the trigger stream when a tenant needs event semantics — a
CDC pipeline, a feature-store update bus, a job-completion notification
fan-out — that has to coexist with the SQL surface the rest of the
platform already uses. Every published event lands as a row in the
topic's backing mutable table; that table is queryable with the same
Flight SQL surface as any other mutable companion table, so ad-hoc
analytics on the event log come for free.

## Goal

Walk through registering one topic for a neutral third-tenant use case
(a small CDC pipeline pulling Postgres change events into a downstream
search index) and publish a batch of events from Rust.

## Setup

Assumes a `JammiSession` whose `JammiConfig.trigger_broker` is left at
its default — the embedded `InMemoryBroker`. Production deployments
swap in `JetStreamBroker` via configuration; the publisher API does
not change.

## Define the topic schema

```rust
use std::sync::Arc;
use arrow_schema::{DataType, Field, Schema};

let schema = Arc::new(Schema::new(vec![
    Field::new("op",         DataType::Utf8,  false),
    Field::new("ts_ms",      DataType::Int64, false),
    Field::new("key",        DataType::Utf8,  false),
    Field::new("after",      DataType::Utf8,  true),
]));
```

The schema is the contract every published batch must satisfy. The
engine reserves the `_offset`, `_row_idx`, and `_produced_at` column
names (all leading-underscore names are reserved); user schemas must
not include them.

## Register the topic

The simplest path is the SQL surface — `session.sql("CREATE TOPIC …")`
parses the same statement Flight SQL clients send across the wire:

```rust,no_run
# use jammi_engine::session::JammiSession;
# async fn ex(session: &JammiSession) -> jammi_engine::error::Result<()> {
session
    .sql(
        "CREATE TOPIC cdc.orders (\
             op TEXT NOT NULL, ts_ms BIGINT NOT NULL, \
             key TEXT NOT NULL, after TEXT) \
         WITH (retention_seconds = '604800')",
    )
    .await?;
# Ok(())
# }
```

The CLI exposes the same shape via `jammi trigger register --name … --schema …`.

```python
import jammi

db = jammi.connect(artifact_dir="/var/lib/jammi")
db.sql(
    """
    CREATE TOPIC cdc.orders (
        op    TEXT NOT NULL,
        ts_ms BIGINT NOT NULL,
        key   TEXT NOT NULL,
        after TEXT
    ) WITH (retention_seconds = '604800')
    """
)
```

For callers that build the topic programmatically (rather than via SQL),
the Rust API surface is equivalent:

```rust,no_run
# use std::collections::BTreeMap;
# use std::sync::Arc;
# use arrow_schema::SchemaRef;
use jammi_engine::trigger::{TopicDefinition, TopicId};

# fn make(schema: SchemaRef) -> TopicDefinition {
let topic = TopicDefinition {
    id: TopicId::new(),
    name: "cdc.orders".into(),
    schema,
    tenant: None,                          // None = global; Some(t) scopes to t
    broker_metadata: BTreeMap::new(),      // driver-specific opts (e.g. retention)
};
# topic
# }
```

The `id` is a UUIDv7 minted at construction — time-ordered so the
catalog index keeps insert locality. The `name` is opaque to the
engine beyond catalog lookup; pick a hierarchical namespace that suits
your platform (e.g. `cdc.orders`, `feature_store.user_features`).

Registration is atomic: the `topics` row, the backing mutable table,
and any broker-side state commit together; nothing lands on failure.

```rust,no_run
# use std::sync::Arc;
# use jammi_engine::catalog::Catalog;
# use jammi_engine::catalog::topic_repo::TopicRepo;
# use jammi_engine::source::mutable::MutableTableRegistry;
# use jammi_engine::trigger::{InMemoryBroker, TopicDefinition, TriggerBroker};
# async fn ex(
#     topic_repo: &TopicRepo,
#     broker: Arc<dyn TriggerBroker>,
#     topic: &TopicDefinition,
# ) -> Result<(), jammi_engine::trigger::TriggerError> {
broker.register_topic(topic).await?;
topic_repo.register_topic(topic).await?;
# Ok(())
# }
```

## Publish a batch

```rust,no_run
# use std::sync::Arc;
# use arrow::array::{Int64Array, RecordBatch, StringArray};
# use arrow_schema::SchemaRef;
# use jammi_engine::trigger::{Publisher, TopicDefinition};
# async fn ex(
#     publisher: &Publisher,
#     topic: &TopicDefinition,
#     schema: SchemaRef,
# ) -> Result<(), jammi_engine::trigger::TriggerError> {
let batch = RecordBatch::try_new(
    schema,
    vec![
        Arc::new(StringArray::from(vec!["c", "u", "d"])),
        Arc::new(Int64Array::from(vec![1700_000_000_000, 1700_000_000_100, 1700_000_000_200])),
        Arc::new(StringArray::from(vec!["order-1", "order-2", "order-3"])),
        Arc::new(StringArray::from(vec![Some("{...}"), Some("{...}"), None])),
    ],
)
.unwrap();
let offset = publisher.publish(topic, batch).await?;
println!("published offset = {}", offset.value());
# Ok(())
# }
```

Python equivalent — `publish_topic` accepts a `pyarrow.Table` via the
Arrow C Stream Interface so the conversion is zero-copy:

```python
import pyarrow as pa

table = pa.table({
    "op":    ["c", "u", "d"],
    "ts_ms": [1700_000_000_000, 1700_000_000_100, 1700_000_000_200],
    "key":   ["order-1", "order-2", "order-3"],
    "after": ["{...}", "{...}", None],
})
offset = db.publish_topic("cdc.orders", batch=table)
print(f"published offset = {offset}")
```

`publish()` validates the batch schema against the topic schema before
opening a transaction. A mismatch returns `BatchSchemaMismatch` and
nothing lands in the backing table.

## What just happened

1. The `Publisher` minted the next monotonic offset for the topic
   (seeded lazily from `MAX(_offset)` on the backing table the first
   time the topic is touched).
2. The augmented batch — user columns plus `_offset`, `_row_idx`, and
   `_produced_at` — was inserted into the topic's backing mutable
   table inside one `CatalogBackend::transaction`. On commit the
   offset advances; on rollback it is reused for the next attempt so
   no gaps appear in the log.
3. The broker received the batch for best-effort fan-out to any live
   subscribers. A broker fan-out failure after commit is logged but
   does *not* fail the publish — subscribers replay missed offsets
   from the backing table on next reconnect.

## See also

- [Subscribe with a SQL Predicate Filter](./subscribe-with-filter.md) —
  open a subscription with a `WHERE`-style predicate and consume the
  matching batches.
- [Replay Events from the Backing Table](./replay-from-backing-table.md)
  — run ad-hoc Flight SQL over the durable event log.
