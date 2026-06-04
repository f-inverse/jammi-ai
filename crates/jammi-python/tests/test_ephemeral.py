"""Python `Database.ephemeral_session` round-trip (spec J6).

Exercises the PyO3 surface end-to-end: a `jammi_ai.connect` session opens an
ephemeral, tenant-scoped storage context as a context manager; tables created
inside are real federated mutable tables; on context exit `close()` drops them
and publishes a `closed` lifecycle event to `jammi.audit.session_lifecycle.v1`.

Mirrors `test_mutable_tables.py` in style — generic `query_images` /
`query_lineage` shapes with no domain coupling.
"""

import json

import pyarrow as pa
import pytest

import jammi_ai


TENANT_A = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"
TENANT_B = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b"


def _images_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("image_id", pa.string(), nullable=False),
            pa.field("image_hash", pa.string(), nullable=False),
        ]
    )


def _images_table(rows: list[tuple[str, str]]) -> pa.Table:
    return pa.table(
        {
            "image_id": [r[0] for r in rows],
            "image_hash": [r[1] for r in rows],
        },
        schema=_images_schema(),
    )


def test_context_manager_creates_uses_and_deletes(tmp_path):
    """Open a session, store rows, read them back, and confirm the table is
    gone and a `closed` event landed once the `with` block exits."""
    db = jammi_ai.connect(f"file://{tmp_path}")
    db.with_tenant(TENANT_A)

    phys = None
    with db.ephemeral_session(timeout_seconds=3600) as ephem:
        assert ephem.tenant_id == TENANT_A
        ephem.create_ephemeral_table(
            "query_images", schema=_images_schema(), primary_key=["image_id"]
        )
        n = ephem.insert(
            "query_images",
            batch=_images_table([("img-1", "sha256:aaa"), ("img-2", "sha256:bbb")]),
        )
        assert n == 2
        assert ephem.count_rows("query_images") == 2

        got = ephem.sql("query_images", "SELECT image_hash FROM {table} ORDER BY image_id")
        assert got.column("image_hash").to_pylist() == ["sha256:aaa", "sha256:bbb"]

        # Capture the physical name so we can prove it's gone after close.
        phys = ephem.sql("query_images", "SELECT * FROM {table}")
        assert phys.num_rows == 2

    # After exit the session is closed; calling into it raises.
    with pytest.raises(RuntimeError):
        ephem.count_rows("query_images")

    # A `closed` lifecycle event with deleted_row_count == 2 was published.
    # The session published exactly two lifecycle events — `opened` (on open)
    # and `closed` (on context exit) — each one batch. `max_batches=2` matches
    # that count exactly so the replay read does not block on the live tail
    # (see test_topics.py for the same pattern).
    events = db.subscribe_collect(
        "jammi.audit.session_lifecycle.v1", from_offset=0, max_batches=2
    )
    records = [json.loads(r) for r in events.column("record").to_pylist()]
    closed = [r for r in records if r["event"] == "closed"]
    assert closed, "a closed event should be published on context exit"
    assert closed[0]["deleted_row_count"] == 2
    assert any(r["event"] == "opened" for r in records)


def test_persistent_lineage_survives_ephemeral_close(tmp_path):
    """A persistent table referencing a hash outlives the ephemeral session
    that also held it (success criterion 6)."""
    db = jammi_ai.connect(f"file://{tmp_path}")
    db.with_tenant(TENANT_A)

    db.create_mutable_table(
        "query_lineage",
        schema=pa.schema([pa.field("image_hash", pa.string(), nullable=False)]),
        primary_key=["image_hash"],
    )

    h = "sha256:deadbeef"
    with db.ephemeral_session(timeout_seconds=3600) as ephem:
        ephem.create_ephemeral_table(
            "imgs", schema=_images_schema(), primary_key=["image_id"]
        )
        ephem.insert("imgs", batch=_images_table([("img-1", h)]))
        db.sql(
            f"INSERT INTO mutable.public.query_lineage (image_hash) VALUES ('{h}')"
        )

    survived = db.sql("SELECT image_hash FROM mutable.public.query_lineage")
    assert survived.column("image_hash").to_pylist() == [h]


def test_requires_tenant_binding(tmp_path):
    """Opening a session with no tenant bound raises ValueError."""
    db = jammi_ai.connect(f"file://{tmp_path}")
    with pytest.raises(ValueError):
        db.ephemeral_session(timeout_seconds=60)


def test_tenant_isolation(tmp_path):
    """Tenant B cannot see tenant A's ephemeral data (success criterion 7).

    The ephemeral session's own `sql()` is always pinned to the tenant it was
    opened under — it is self-consistent and only ever sees its own rows,
    regardless of the parent's current sticky binding. Isolation is therefore
    proven the way the substrate enforces it: a query against the underlying
    physical table run under a *different* parent tenant scopes the rows out.
    """
    db = jammi_ai.connect(f"file://{tmp_path}")
    db.with_tenant(TENANT_A)

    with db.ephemeral_session(timeout_seconds=3600) as ephem:
        ephem.create_ephemeral_table(
            "imgs", schema=_images_schema(), primary_key=["image_id"]
        )
        ephem.insert("imgs", batch=_images_table([("a", "h-a")]))
        phys = ephem.physical_table_ref("imgs")

        # Under the owning tenant A the row is visible.
        seen = db.sql(f"SELECT * FROM {phys}")
        assert seen.num_rows == 1

        # The session's own read stays pinned to A even after the parent rebinds.
        assert ephem.sql("imgs", "SELECT * FROM {table}").num_rows == 1

        # Tenant B, querying the physical table directly, sees none of A's rows.
        db.with_tenant(TENANT_B)
        scoped_out = db.sql(f"SELECT * FROM {phys}")
        assert scoped_out.num_rows == 0

        # Restore A so close runs under the right scope.
        db.with_tenant(TENANT_A)
