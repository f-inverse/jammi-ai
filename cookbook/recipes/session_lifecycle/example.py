"""Ephemeral session storage — auto-deleted on session end.

An ephemeral session is a tenant-scoped storage context whose tables are
deleted automatically when the session ends: on explicit `close`, on context-
manager exit, or when the timeout scanner force-closes it. Every transition
publishes to the `jammi.audit.session_lifecycle.v1` trigger topic, so an audit-
log aggregator can prove the deletion happened.

Use it for sensitive transient data — uploaded images, derived embeddings,
draft inputs — that must not outlive the request that produced it. Put long-
lived data (the audit record, the persistent corpus) in ordinary tables; the
ephemeral session holds only the throwaway working set.

Run from the repo root:  python cookbook/recipes/session_lifecycle/example.py
"""

from __future__ import annotations

import hashlib
import tempfile

import pyarrow as pa

import jammi_ai

TENANT = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"


def _images_table(rows: list[tuple[str, str]]) -> pa.Table:
    return pa.table(
        {
            "image_id": [r[0] for r in rows],
            "image_hash": [r[1] for r in rows],
        },
        schema=pa.schema(
            [
                pa.field("image_id", pa.string(), nullable=False),
                pa.field("image_hash", pa.string(), nullable=False),
            ]
        ),
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = jammi_ai.connect(f"file://{tmp}")
        db.set_tenant(TENANT)

        # A persistent table that will keep the *hash* lineage after the raw
        # working data is deleted — what NOT to put in ephemeral storage.
        db.create_mutable_table(
            "query_lineage",
            schema=pa.schema([pa.field("image_hash", pa.string(), nullable=False)]),
            primary_key=["image_hash"],
        )

        # Hash two "uploaded" images. Only the hashes are durable.
        uploads = {"img-1": b"...image one bytes...", "img-2": b"...image two bytes..."}
        hashes = {
            iid: "sha256:" + hashlib.sha256(data).hexdigest()
            for iid, data in uploads.items()
        }

        # 1. Open an ephemeral session as a context manager. Tables created
        #    inside are deleted on exit.
        with db.ephemeral_session(timeout_seconds=3600) as ephem:
            ephem.create_ephemeral_table(
                "query_images",
                schema=pa.schema(
                    [
                        pa.field("image_id", pa.string(), nullable=False),
                        pa.field("image_hash", pa.string(), nullable=False),
                    ]
                ),
                primary_key=["image_id"],
            )
            rows = [(iid, h) for iid, h in hashes.items()]
            inserted = ephem.insert("query_images", batch=_images_table(rows))
            assert inserted == 2, "two rows stored in the ephemeral table"
            assert ephem.count_rows("query_images") == 2

            # 2. Use the ephemeral data (here: read the hashes back), then write
            #    only the hash lineage to the PERSISTENT table — before close,
            #    while the ephemeral data still exists.
            stored = ephem.sql("query_images", "SELECT image_hash FROM {table}")
            for h in stored.column("image_hash").to_pylist():
                db.sql(
                    f"INSERT INTO mutable.public.query_lineage (image_hash) VALUES ('{h}')"
                )
            print("ephemeral rows during session:", ephem.count_rows("query_images"))
        # 3. Context exit called close(): every ephemeral table is dropped and a
        #    `closed` event was published to jammi.audit.session_lifecycle.v1.

        # 4. The persistent lineage survives; it references the hashes, never the
        #    deleted working data.
        lineage = db.sql("SELECT image_hash FROM mutable.public.query_lineage")
        assert lineage.num_rows == 2, "hash lineage persists after session close"
        print("persistent lineage rows after close:", lineage.num_rows)

        # 5. The lifecycle stream carries the proof-of-deletion events. Replay
        #    from offset 0 and confirm an `opened` and a `closed` event landed.
        #    The session published exactly two events (opened + closed), each one
        #    batch; `max_batches=2` matches that count so the replay read does not
        #    block waiting on the live tail.
        events = db.subscribe_collect(
            "jammi.audit.session_lifecycle.v1", from_offset=0, max_batches=2
        )
        import json

        kinds = [json.loads(r)["event"] for r in events.column("record").to_pylist()]
        assert "opened" in kinds, "opened event published"
        assert "closed" in kinds, "closed event published"
        closed = next(
            json.loads(r)
            for r in events.column("record").to_pylist()
            if json.loads(r)["event"] == "closed"
        )
        assert closed["deleted_row_count"] == 2, "closed event reports deleted rows"
        print("lifecycle events:", kinds)

        print("session_lifecycle recipe OK")


if __name__ == "__main__":
    main()
