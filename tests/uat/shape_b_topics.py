"""UAT shape-B: drive the topic-registration primitive from a Python client.

Registers a trigger-stream topic with broker metadata, confirms the
topic catalog returns it via `list_topics`, then drops it. The publish
and subscribe side of the trigger primitive is exercised in
`shape_b_trigger.py`.

Run with `python3 tests/uat/shape_b_topics.py`. Exits 0 on success.
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

        topic_id = db.register_topic(
            "events.demo",
            schema=_events_schema(),
            broker_metadata={"retention_seconds": "3600"},
        )
        assert len(topic_id) > 0, "register_topic returned empty id"

        topics = db.list_topics()
        assert "events.demo" in topics, f"events.demo missing from {topics}"

        db.drop_topic("events.demo")
        topics = db.list_topics()
        assert "events.demo" not in topics, f"events.demo persisted after drop"

        # Idempotent drop with if_exists.
        db.drop_topic("events.demo", if_exists=True)

        # Drop on a missing topic without if_exists must raise.
        try:
            db.drop_topic("never.registered")
        except (ValueError, RuntimeError) as exc:
            assert "never.registered" in str(exc), (
                f"drop-missing error lost topic name: {exc}"
            )
        else:
            raise AssertionError("drop-missing must raise")

    print("shape_b_topics: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
