"""Live remote round-trip for the mutable-table + topic + pub/sub verbs.

Stands up a real CPU `jammi-server` (no GPU — these are DML/streaming verbs) and
drives the two cookbook-unblocking paths through the pure-Python
`RemoteDatabase`, asserting the remote results equal the embedded engine's:

  * **C1 (mutable table):** `create_mutable_table` → write rows via `sql()` (the
    Flight SQL data-plane DML path) → read them back / federated JOIN with a
    registered source → `drop_mutable_table`.
  * **C2 (topic pub/sub):** `register_topic` → `publish_topic(batch)` →
    `subscribe_collect(predicate, from_offset)` returns the published rows →
    backing-table replay via `sql()` over the topic's backing table.

Each path runs the SAME calls against an embedded `jammi_ai.Database` and asserts
the two transports agree (parity). A two-tenant isolation assertion confirms each
verb rides the session's `jammi-session-id` scope — a mutable table created under
tenant A is invisible to tenant B.

Gated, not hermetic: the test needs a built server binary, so it is skipped
unless `JAMMI_SERVER_BIN` points at a `jammi-server` executable. CI's
python-test job sets it after building the binary; a bare `pytest` skips it. The
embedded engine (`jammi_ai`) must also be importable (the parity peer).
"""

from __future__ import annotations

import os

import pyarrow as pa
import pytest

jammi_ai = pytest.importorskip("jammi_ai")
import jammi_client  # noqa: E402

SERVER_BIN = os.environ.get("JAMMI_SERVER_BIN")

pytestmark = pytest.mark.skipif(
    not SERVER_BIN or not os.path.exists(SERVER_BIN),
    reason="JAMMI_SERVER_BIN not set to a built jammi-server binary",
)

# Two syntactically valid tenant UUIDs for the isolation assertion.
TENANT_A = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"
TENANT_B = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b"


def _dim_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("sku", pa.string(), nullable=False),
            pa.field("name", pa.string(), nullable=False),
            pa.field("price_cents", pa.int64(), nullable=False),
        ]
    )


def _events_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("event_id", pa.int64(), nullable=False),
            pa.field("kind", pa.string(), nullable=False),
        ]
    )


def test_c1_mutable_table_round_trip_matches_embedded(live_server, tmp_path):
    """C1: create a mutable table, write rows via Flight SQL DML, read them back,
    then drop — remote results equal the embedded engine's."""
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    try:
        for db in (remote, embedded):
            db.create_mutable_table(
                "dim_products",
                schema=_dim_schema(),
                primary_key=["sku"],
            )
            # The mutable table appears in the registry introspection.
            ids = [t["id"] for t in db.list_mutable_tables()]
            assert "dim_products" in ids

            # Write rows through the Flight SQL / engine DML path (the data plane).
            db.sql(
                "INSERT INTO mutable.public.dim_products (sku, name, price_cents) "
                "VALUES ('a', 'Widget', 100), ('b', 'Gadget', 250)"
            )

        remote_rows = remote.sql(
            "SELECT sku, name, price_cents FROM mutable.public.dim_products ORDER BY sku"
        )
        embedded_rows = embedded.sql(
            "SELECT sku, name, price_cents FROM mutable.public.dim_products ORDER BY sku"
        )
        assert remote_rows.to_pydict() == embedded_rows.to_pydict()
        assert remote_rows.num_rows == 2

        # A federated JOIN of the mutable table against itself (an aggregate over
        # the same data) exercises the table as a first-class query participant.
        remote_join = remote.sql(
            "SELECT count(*) AS n, sum(price_cents) AS total "
            "FROM mutable.public.dim_products"
        )
        embedded_join = embedded.sql(
            "SELECT count(*) AS n, sum(price_cents) AS total "
            "FROM mutable.public.dim_products"
        )
        assert remote_join.to_pydict() == embedded_join.to_pydict()

        for db in (remote, embedded):
            db.drop_mutable_table("dim_products")
            assert "dim_products" not in [t["id"] for t in db.list_mutable_tables()]
            # `if_exists` makes a second drop a no-op on both transports.
            db.drop_mutable_table("dim_products", if_exists=True)
    finally:
        # The embedded engine releases its resources on drop (RAII); only the
        # remote client holds a gRPC channel that needs an explicit close.
        remote.close()


def test_c2_topic_pub_sub_round_trip_matches_embedded(live_server, tmp_path):
    """C2: register a topic, publish a batch, then `subscribe_collect` with a
    predicate + from_offset returns the published rows — remote equals embedded,
    and the topic's backing table replays the same rows via `sql()`."""
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    try:
        batch = pa.table(
            {
                "event_id": pa.array([1, 2, 3], type=pa.int64()),
                "kind": pa.array(["click", "view", "click"], type=pa.string()),
            },
            schema=_events_schema(),
        )

        collected = {}
        for name, db in (("remote", remote), ("embedded", embedded)):
            db.register_topic("events.demo", schema=_events_schema())
            assert "events.demo" in db.list_topics()

            offset = db.publish_topic("events.demo", batch=batch)
            assert offset == 0  # first publish on a fresh topic

            # Replay from offset 0 with a predicate; `max_batches=1` matches the
            # single publish so the bounded collect does not race the live tail.
            got = db.subscribe_collect(
                "events.demo",
                predicate="kind = 'click'",
                from_offset=0,
                max_batches=1,
            )
            collected[name] = got

        # The predicate filtered to the two 'click' rows on both transports.
        assert collected["remote"].to_pydict() == collected["embedded"].to_pydict()
        kinds = collected["remote"].column("kind").to_pylist()
        assert kinds == ["click", "click"]

        # Backing-table replay: an unfiltered `subscribe_collect(from_offset=0)`
        # drains the topic's durable backing table (the persisted event log behind
        # the stream) and returns every published row — identical across
        # transports. This is the replay half of the replay+live-tail join.
        remote_replay = remote.subscribe_collect(
            "events.demo", from_offset=0, max_batches=1
        )
        embedded_replay = embedded.subscribe_collect(
            "events.demo", from_offset=0, max_batches=1
        )
        assert remote_replay.to_pydict() == embedded_replay.to_pydict()
        assert remote_replay.num_rows == 3

        for db in (remote, embedded):
            db.drop_topic("events.demo")
            assert "events.demo" not in db.list_topics()
            db.drop_topic("events.demo", if_exists=True)  # no-op on both
    finally:
        # The embedded engine releases its resources on drop (RAII); only the
        # remote client holds a gRPC channel that needs an explicit close.
        remote.close()


def test_mutable_table_tenant_isolation_over_the_wire(live_server):
    """A mutable table created under tenant A is invisible to tenant B — proving
    every new verb rides the session's tenant scope (a verb missing
    `metadata=self._metadata` would silently run unscoped and leak across
    tenants). Two separate connections carry two distinct session ids."""
    db_a = jammi_client.connect(live_server)
    db_b = jammi_client.connect(live_server)
    try:
        db_a.set_tenant(TENANT_A)
        db_b.set_tenant(TENANT_B)

        db_a.create_mutable_table(
            "scoped_dim",
            schema=_dim_schema(),
            primary_key=["sku"],
        )

        # Tenant A sees it; tenant B does not.
        assert "scoped_dim" in [t["id"] for t in db_a.list_mutable_tables()]
        assert "scoped_dim" not in [t["id"] for t in db_b.list_mutable_tables()]

        db_a.drop_mutable_table("scoped_dim")
    finally:
        db_a.close()
        db_b.close()


def test_tenant_scope_scopes_and_restores_over_the_wire(live_server):
    """`with db.tenant_scope(t)` scopes server-side reads to `t` for the block and
    restores the prior tenant on exit — over gRPC. The remote tenant lives
    server-side (keyed by session id); the client captures the prior tenant via
    `GetTenant` on entry and rebinds it on exit. Nesting restores the outer tenant
    on inner exit, not a blind clear. Registry isolation (`list_mutable_tables`,
    proven session-scoped by the isolation test above) is the observable."""
    db = jammi_client.connect(live_server)
    other = jammi_client.connect(live_server)
    try:
        # `other` registers a B-only table the scoped `db` must not see under A.
        other.set_tenant(TENANT_B)
        other.create_mutable_table(
            "b_only", schema=_dim_schema(), primary_key=["sku"]
        )

        assert db.tenant() is None  # starts unscoped

        with db.tenant_scope(TENANT_A):
            assert db.tenant() == TENANT_A
            db.create_mutable_table(
                "a_only", schema=_dim_schema(), primary_key=["sku"]
            )
            ids = {t["id"] for t in db.list_mutable_tables()}
            assert "a_only" in ids and "b_only" not in ids  # scoped to A

            with db.tenant_scope(TENANT_B):
                assert db.tenant() == TENANT_B
                inner = {t["id"] for t in db.list_mutable_tables()}
                assert "b_only" in inner and "a_only" not in inner
            # Inner exit restores A, not unscoped.
            assert db.tenant() == TENANT_A
            assert "a_only" in {t["id"] for t in db.list_mutable_tables()}

            db.drop_mutable_table("a_only")

        # Outer exit restores the prior (unscoped) scope.
        assert db.tenant() is None

        other.drop_mutable_table("b_only")
    finally:
        db.close()
        other.close()
