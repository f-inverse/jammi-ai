# Trigger streams

End-to-end publish + subscribe on a Jammi topic, plus the registration
and listing surface. Uses the embedded in-process broker — no external
broker needed.

**When to use this pattern.** You need a low-friction event bus inside
your application — for fan-out to downstream consumers, fan-in from
batch jobs, or replay-from-offset semantics — without bringing up a
separate message broker. The same surface scales out across replicas on
the Postgres broker (`[broker.postgres]`) by a config change at deploy
time.

## API surface exercised

- `Session.register_topic(name, *, schema)`
- `Session.list_topics()`
- `Session.publish_topic(name, *, batch)` — returns the assigned offset
- `Session.subscribe_collect(name, *, from_offset)`
- `Session.drop_topic(name, *, if_exists=False)`

`subscribe_collect` drains the backing table from `from_offset` and
returns; `replay_only=False` with `max_batches` follows the live tail
instead, returning once that many batches arrive.

## Run it

```bash
python cookbook/recipes/trigger_streams/example.py
```

It prints the topic list, the batch read back, and the error a missing topic raises.
