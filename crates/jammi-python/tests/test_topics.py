"""Python `Database.register_topic` / `drop_topic` round-trip.

Exercises the PyO3 surface end-to-end: a `jammi.connect` session
registers a trigger-stream topic; the catalog row + backing mutable
table + broker driver entry commit atomically; publish + subscribe land
matching rows back through the Python binding.

Mirrors `test_tenant.py` in style; uses generic schema shapes
(`events.demo`, `cdc.orders`) with no domain coupling.
"""

import pyarrow as pa
import pytest

import jammi


TENANT_A = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"
TENANT_B = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b"


def _events_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("event_id", pa.int64(), nullable=False),
            pa.field("payload", pa.string(), nullable=False),
        ]
    )


def _orders_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("order_id", pa.int64(), nullable=False),
            pa.field("amount", pa.float64(), nullable=False),
        ]
    )


def test_register_publish_subscribe_round_trips(tmp_path):
    """Register `events.demo`, publish one batch, read it back via
    `subscribe_collect`."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    db.register_topic("events.demo", schema=_events_schema())

    batch = pa.table(
        {
            "event_id": pa.array([1, 2, 3], type=pa.int64()),
            "payload": pa.array(["a", "b", "c"], type=pa.string()),
        },
        schema=_events_schema(),
    )
    offset = db.publish_topic("events.demo", batch=batch)
    # The broker assigns offsets sequentially per topic starting at 0,
    # so the first publish on a fresh topic returns offset 0.
    assert offset == 0

    # Replay the one published batch, then exit before the live tail blocks.
    # `from_offset=0` triggers backing-table replay; `max_batches=1` matches
    # the publish count exactly so the read does not race the live broker.
    collected = db.subscribe_collect(
        "events.demo", from_offset=0, max_batches=1
    )
    event_ids = collected.column("event_id").to_pylist()
    payloads = collected.column("payload").to_pylist()
    assert event_ids == [1, 2, 3]
    assert payloads == ["a", "b", "c"]


def test_register_with_broker_metadata(tmp_path):
    """`broker_metadata` is opaque driver config; the topic catalog round-
    trips it as-is and `list_topics` includes the registered topic."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    db.register_topic(
        "cdc.orders",
        schema=_orders_schema(),
        broker_metadata={"retention_seconds": "3600"},
    )

    topics = db.list_topics()
    assert "cdc.orders" in topics


def test_register_inherits_session_tenant(tmp_path):
    """Tenant inheritance on register + tenant-scoped listing on the same
    session: bound to tenant A, `events.demo` appears in `list_topics`;
    re-bound to tenant B, the same call excludes the A-scoped topic;
    re-bound back to A, the topic is visible again."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    db.with_tenant(TENANT_A)
    db.register_topic("events.demo", schema=_events_schema())

    assert "events.demo" in db.list_topics()

    db.with_tenant(TENANT_B)
    assert "events.demo" not in db.list_topics()

    db.with_tenant(TENANT_A)
    assert "events.demo" in db.list_topics()


def test_drop_missing_without_if_exists_raises(tmp_path):
    """`drop_topic` over an unknown name raises — the binding rejects with
    `ValueError`."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    with pytest.raises((ValueError, RuntimeError)) as info:
        db.drop_topic("never.registered")
    msg = str(info.value).lower()
    assert "not found" in msg, (
        f"expected the typed not-found message, got: {info.value}"
    )


def test_drop_missing_with_if_exists_succeeds(tmp_path):
    """`drop_topic(if_exists=True)` over an unknown name is a no-op."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    db.drop_topic("never.registered", if_exists=True)


def test_register_rejects_unsupported_schema_type(tmp_path):
    """A topic schema with a column dtype outside the catalog encoder's
    closed set (e.g. `Timestamp`) raises with `UnsupportedSchemaType` —
    the engine's typed error from the band-aid fix surfaces verbatim."""
    db = jammi.connect(artifact_dir=str(tmp_path))
    schema_with_timestamp = pa.schema(
        [
            pa.field("event_id", pa.int64(), nullable=False),
            pa.field("when", pa.timestamp("us"), nullable=False),
        ]
    )
    with pytest.raises((ValueError, RuntimeError)) as info:
        db.register_topic("events.demo", schema=schema_with_timestamp)
    msg = str(info.value).lower()
    assert "unsupported" in msg and "when" in msg, (
        f"expected the typed UnsupportedSchemaType message naming the column, got: {info.value}"
    )
