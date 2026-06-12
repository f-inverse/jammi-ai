"""Live remote round-trip for the evidence-channel registry verbs.

Stands up a real CPU `jammi-server` (catalog-plane verbs, no GPU) and drives the
channel family through the pure-Python `RemoteDatabase`, asserting the remote
results equal the embedded engine's:

  * **register → list:** `register_channel` then `list_channels` on the remote
    equals the embedded engine's `list_channels` (the same dict shape, including
    the global seed channels both transports carry).
  * **append → relist:** `add_channel_columns` appends in order; a re-`list`
    shows the appended columns after the originals on both transports.
  * **error parity:** re-registering an id raises on BOTH transports
    ("already exists"); redeclaring a column with a DIFFERENT dtype raises the
    append-only violation on both; redeclaring with the SAME dtype raises
    "already declared" on both (the engine rejects, not no-ops).
  * **tenant scoping over the wire:** a channel registered under tenant A is
    invisible to tenant B (and B may register the same id without collision);
    an unbound connection sees only the global (NULL-tenant) seed channels —
    the #170 tenant-qualification property, now driven through the client.

The `live_server` fixture is module-scoped, so its catalog state persists across
tests in this module. Each parity test therefore scopes BOTH transports to a
FRESH per-test tenant (a unique UUID); a channel registered under that tenant is
isolated from every other test's registrations, so a full `list_channels`
equality between the two transports is well-defined (each sees that tenant's own
channels plus the shared global seeds, nothing leaked from a sibling test). The
embedded peer binds the SAME tenant so the two namespaces line up.

Gated, not hermetic: the test needs a built server binary, so it is skipped
unless `JAMMI_SERVER_BIN` points at a `jammi-server` executable. CI's
python-test job sets it after building the binary; a bare `pytest` skips it. The
embedded engine (`jammi_ai`) must also be importable (the parity peer).
"""

from __future__ import annotations

import os
import uuid

import pytest

jammi_ai = pytest.importorskip("jammi_ai")
import jammi_client  # noqa: E402

SERVER_BIN = os.environ.get("JAMMI_SERVER_BIN")

pytestmark = pytest.mark.skipif(
    not SERVER_BIN or not os.path.exists(SERVER_BIN),
    reason="JAMMI_SERVER_BIN not set to a built jammi-server binary",
)


def _fresh_tenant() -> str:
    """A unique tenant UUID, isolating one test's registrations on the shared
    module-scoped server from every sibling test's."""
    return str(uuid.uuid4())


def test_register_then_list_matches_embedded(live_server, tmp_path):
    """Register a channel on both transports, then `list_channels` on the remote
    equals the embedded engine's — the same dict shape, the global seed channels
    plus the freshly-registered one, ordered by `(priority, channel_id)`."""
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    tenant = _fresh_tenant()
    try:
        for db in (remote, embedded):
            db.set_tenant(tenant)
            db.register_channel(
                "scored_by",
                priority=50,
                columns=[("score", "Float64"), ("model", "Utf8")],
            )

        remote_list = remote.list_channels()
        embedded_list = embedded.list_channels()
        assert remote_list == embedded_list

        # The freshly-registered channel is present with the exact declared shape.
        scored = next(c for c in remote_list if c["channel_id"] == "scored_by")
        assert scored == {
            "channel_id": "scored_by",
            "priority": 50,
            "columns": [
                {"name": "score", "data_type": "Float64"},
                {"name": "model", "data_type": "Utf8"},
            ],
        }
        # The global seed channels both transports carry are visible too.
        names = {c["channel_id"] for c in remote_list}
        assert {"vector", "inference"} <= names
    finally:
        remote.close()


def test_add_columns_appends_in_order_matches_embedded(live_server, tmp_path):
    """`add_channel_columns` appends new columns after the originals, in
    declaration order — a re-`list` agrees across transports."""
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    tenant = _fresh_tenant()
    try:
        for db in (remote, embedded):
            db.set_tenant(tenant)
            db.register_channel(
                "annotated_by",
                priority=10,
                columns=[("label", "Utf8")],
            )
            db.add_channel_columns(
                "annotated_by",
                columns=[("confidence", "Float32"), ("rank", "Int32")],
            )

        remote_list = remote.list_channels()
        assert remote_list == embedded.list_channels()

        annotated = next(
            c for c in remote_list if c["channel_id"] == "annotated_by"
        )
        assert annotated["columns"] == [
            {"name": "label", "data_type": "Utf8"},
            {"name": "confidence", "data_type": "Float32"},
            {"name": "rank", "data_type": "Int32"},
        ]
    finally:
        remote.close()


def test_reregister_same_id_rejected_on_both_transports(live_server, tmp_path):
    """Re-registering an already-registered channel id raises on BOTH transports
    — the per-tenant uniqueness "already exists" path."""
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    tenant = _fresh_tenant()
    try:
        for db in (remote, embedded):
            db.set_tenant(tenant)
            db.register_channel(
                "dup_chan", priority=1, columns=[("a", "Int64")]
            )

        with pytest.raises(Exception) as embedded_err:
            embedded.register_channel(
                "dup_chan", priority=1, columns=[("a", "Int64")]
            )
        with pytest.raises(Exception) as remote_err:
            remote.register_channel(
                "dup_chan", priority=1, columns=[("a", "Int64")]
            )
        assert "already exists" in str(embedded_err.value)
        assert "already exists" in str(remote_err.value)
    finally:
        remote.close()


def test_redeclare_column_different_dtype_rejected_on_both(live_server, tmp_path):
    """Redeclaring an existing column with a DIFFERENT dtype raises the
    append-only violation on BOTH transports."""
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    tenant = _fresh_tenant()
    try:
        for db in (remote, embedded):
            db.set_tenant(tenant)
            db.register_channel(
                "typed_chan", priority=1, columns=[("v", "Float32")]
            )

        with pytest.raises(Exception) as embedded_err:
            embedded.add_channel_columns(
                "typed_chan", columns=[("v", "Float64")]
            )
        with pytest.raises(Exception) as remote_err:
            remote.add_channel_columns(
                "typed_chan", columns=[("v", "Float64")]
            )
        # The engine's append-only message names the prior and rejected dtypes.
        assert "cannot redeclare" in str(embedded_err.value)
        assert "cannot redeclare" in str(remote_err.value)
    finally:
        remote.close()


def test_redeclare_column_same_dtype_rejected_on_both(live_server, tmp_path):
    """Redeclaring an existing column with the SAME dtype raises "already
    declared" on BOTH transports — the engine rejects an idempotent redeclare,
    it does not silently no-op, and the client mirrors that exactly."""
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    tenant = _fresh_tenant()
    try:
        for db in (remote, embedded):
            db.set_tenant(tenant)
            db.register_channel(
                "same_chan", priority=1, columns=[("v", "Int64")]
            )

        with pytest.raises(Exception) as embedded_err:
            embedded.add_channel_columns("same_chan", columns=[("v", "Int64")])
        with pytest.raises(Exception) as remote_err:
            remote.add_channel_columns("same_chan", columns=[("v", "Int64")])
        assert "already declared" in str(embedded_err.value)
        assert "already declared" in str(remote_err.value)
    finally:
        remote.close()


def test_tenant_scoping_over_the_wire_matches_embedded(live_server, tmp_path):
    """The #170 tenant-qualification property, driven through the client:
    a channel registered under tenant A is invisible to tenant B; B may register
    the same id without collision; an unbound connection sees only the global
    seed channels. The remote client propagates tenant scope identically to the
    embedded engine — asserted by full `list_channels` parity at each scope.

    Two FRESH tenants isolate this test's registrations from the module-scoped
    server's accumulated state; the unbound assertion checks only that the
    tenant's channel is absent and the seeds are present (the unbound list also
    carries sibling tests' global-tenant rows, which is irrelevant here)."""
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    tenant_a = _fresh_tenant()
    tenant_b = _fresh_tenant()
    try:
        # Under tenant A, register "x_chan".
        for db in (remote, embedded):
            with db.tenant_scope(tenant_a):
                db.register_channel(
                    "x_chan", priority=7, columns=[("w", "Float32")]
                )

        # Tenant A sees its channel plus the global seeds; remote == embedded.
        with remote.tenant_scope(tenant_a):
            remote_a = remote.list_channels()
        with embedded.tenant_scope(tenant_a):
            embedded_a = embedded.list_channels()
        assert remote_a == embedded_a
        a_names = {c["channel_id"] for c in remote_a}
        assert "x_chan" in a_names
        assert {"vector", "inference"} <= a_names

        # Tenant B does NOT see A's "x_chan" — separate per-tenant namespaces.
        with remote.tenant_scope(tenant_b):
            remote_b_before = remote.list_channels()
        with embedded.tenant_scope(tenant_b):
            embedded_b_before = embedded.list_channels()
        assert remote_b_before == embedded_b_before
        assert "x_chan" not in {c["channel_id"] for c in remote_b_before}

        # B may register its OWN "x_chan" without colliding with A's — and the
        # two namespaces stay distinct (B's columns are B's, not A's).
        for db in (remote, embedded):
            with db.tenant_scope(tenant_b):
                db.register_channel(
                    "x_chan", priority=3, columns=[("note", "Utf8")]
                )

        with remote.tenant_scope(tenant_b):
            remote_b = remote.list_channels()
        with embedded.tenant_scope(tenant_b):
            embedded_b = embedded.list_channels()
        assert remote_b == embedded_b
        b_x = next(c for c in remote_b if c["channel_id"] == "x_chan")
        assert b_x == {
            "channel_id": "x_chan",
            "priority": 3,
            "columns": [{"name": "note", "data_type": "Utf8"}],
        }

        # A's "x_chan" is unchanged by B's registration — own-namespace isolation.
        with remote.tenant_scope(tenant_a):
            a_x = next(
                c for c in remote.list_channels() if c["channel_id"] == "x_chan"
            )
        assert a_x == {
            "channel_id": "x_chan",
            "priority": 7,
            "columns": [{"name": "w", "data_type": "Float32"}],
        }

        # An unbound connection sees only the global (NULL-tenant) channels —
        # never a tenant's "x_chan". (Both transports' unbound lists carry the
        # global seeds; the remote's also carries sibling tests' global rows, so
        # this asserts the tenant-scoped row's ABSENCE, not full-list equality.)
        remote_unbound = remote.list_channels()
        unbound_names = {c["channel_id"] for c in remote_unbound}
        assert "x_chan" not in unbound_names
        assert {"vector", "inference"} <= unbound_names
    finally:
        remote.close()
