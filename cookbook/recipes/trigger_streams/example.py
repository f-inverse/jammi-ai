"""Publish + subscribe on a Jammi topic via the in-process broker.

Run with `python cookbook/recipes/trigger_streams/example.py`. Exits 0
on success.
"""

from __future__ import annotations

import tempfile

import pyarrow as pa

import jammi_ai


def events_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("event_id", pa.int64(), nullable=False),
            pa.field("payload", pa.string(), nullable=False),
        ]
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db = jammi_ai.connect(f"file://{tmp}")

        # 1. Register a topic with a typed schema. `broker_metadata`
        #    flows through to the broker driver — the in-process broker
        #    accepts arbitrary string-keyed metadata.
        topic_id = db.register_topic(
            "events.demo",
            schema=events_schema(),
            broker_metadata={"retention_seconds": "3600"},
        )
        assert topic_id, "register_topic returned empty id"

        # 2. The catalog now lists the topic for the current tenant.
        topics = db.list_topics()
        assert "events.demo" in topics, f"events.demo missing from {topics}"

        # 3. Publish one batch. The broker assigns sequential offsets per
        #    topic, starting at 0 for a fresh topic.
        batch = pa.table(
            {
                "event_id": pa.array([1, 2, 3], type=pa.int64()),
                "payload": pa.array(["alpha", "beta", "gamma"], type=pa.string()),
            },
            schema=events_schema(),
        )
        offset = db.publish_topic("events.demo", batch=batch)
        assert offset == 0, f"expected offset 0, got {offset}"

        # 4. Subscribe from offset 0 — drives the backing-table replay
        #    path. `max_batches=1` matches the published batch count so
        #    the call returns immediately without racing the live tail.
        collected = db.subscribe_collect(
            "events.demo", from_offset=0, max_batches=1
        )
        assert collected.column("event_id").to_pylist() == [1, 2, 3]
        assert collected.column("payload").to_pylist() == ["alpha", "beta", "gamma"]

        # 5. Drop the topic and confirm it leaves the catalog.
        db.drop_topic("events.demo")
        topics = db.list_topics()
        assert "events.demo" not in topics, "events.demo persisted after drop"

        # 6. Idempotent drop — `if_exists=True` swallows the missing case.
        db.drop_topic("events.demo", if_exists=True)

        # 7. Strict drop on a missing topic raises with a useful message.
        try:
            db.drop_topic("never.registered")
        except (ValueError, RuntimeError) as exc:
            assert "never.registered" in str(exc), (
                f"drop-missing error lost topic name: {exc}"
            )
        else:
            raise AssertionError("drop-missing must raise")

    print("trigger_streams: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
