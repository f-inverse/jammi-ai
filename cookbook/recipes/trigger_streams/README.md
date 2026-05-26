# Trigger streams

End-to-end publish + subscribe on a Jammi topic, plus the registration
and listing surface. Uses the embedded in-process broker — no NATS or
external broker needed.

**When to use this pattern.** You need a low-friction event bus inside
your application — for fan-out to downstream consumers, fan-in from
batch jobs, or replay-from-offset semantics — without bringing up Kafka
or NATS in dev/test. The same surface scales out to NATS JetStream by
flipping a config flag at deploy time.

## What `example.py` does

1. Connects to a temporary artifact dir
2. Registers a topic `events.demo` with a typed schema and broker
   metadata
3. Confirms `list_topics()` returns the new topic
4. Publishes a 3-row batch through `publish_topic` — captures the
   broker-assigned offset
5. Subscribes from `from_offset=0` and round-trips the same rows back
6. Drops the topic, confirms it's gone from `list_topics()`
7. Demonstrates idempotent `drop_topic(..., if_exists=True)` and
   strict-mode failure when dropping a missing topic

## API surface exercised

- `Database.register_topic(name, *, schema, broker_metadata=None)`
- `Database.list_topics()`
- `Database.publish_topic(name, *, batch)` — returns the assigned offset
- `Database.subscribe_collect(name, *, from_offset, max_batches)`
- `Database.drop_topic(name, *, if_exists=False)`

The `subscribe_collect` path drives the replay-from-backing-table flow
when `from_offset=0`; the live-tail flow is exercised in the broker
integration suite.

## Run it

```bash
python cookbook/recipes/trigger_streams/example.py
```

Exits 0 on success, prints `trigger_streams: OK` on the last line.
