"""UAT shape-B: drive the trigger publish+subscribe path from a Python client.

Registers a topic, publishes one batch, and reads it back via
`subscribe_collect` with `from_offset=0` (the backing-table replay
path). Validates that the publish/subscribe round-trip ships through
the embedded broker.

Run with `python3 tests/uat/shape_b_trigger.py`. Exits 0 on success.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pyarrow as pa

import jammi_ai


def _events_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("event_id", pa.int64(), nullable=False),
            pa.field("payload", pa.string(), nullable=False),
        ]
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = str(Path(tmp))
        db = jammi_ai.connect(artifact_dir=artifact_dir)

        db.register_topic("events.demo", schema=_events_schema())

        batch = pa.table(
            {
                "event_id": pa.array([1, 2, 3], type=pa.int64()),
                "payload": pa.array(["a", "b", "c"], type=pa.string()),
            },
            schema=_events_schema(),
        )
        offset = db.publish_topic("events.demo", batch=batch)
        # Broker assigns sequential offsets per topic starting at 0; the
        # first publish on a fresh topic returns 0.
        assert offset == 0, f"expected offset 0, got {offset}"

        # `from_offset=0` drives the backing-table replay path and
        # `max_batches=1` matches the published batch count so the
        # subscribe does not race the live tail.
        collected = db.subscribe_collect(
            "events.demo", from_offset=0, max_batches=1
        )
        assert collected.column("event_id").to_pylist() == [1, 2, 3]
        assert collected.column("payload").to_pylist() == ["a", "b", "c"]

        db.drop_topic("events.demo")

    print("shape_b_trigger: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
