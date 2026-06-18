"""Hermetic embedded-live checks for the provenance-channel vertical (chapter 14).

These run the channel sequence LIVE against the embedded ``jammi_ai`` engine each
time (a fresh ``file://`` temp catalog, no server, no GPU) and assert the channel
family's load-bearing facts — the embedded-live half that catches an embedded
regression on every PR (the cross-transport parity is the emit-side check):

* **register → list round-trip:** a registered channel reappears in
  ``list_channels`` with its exact declared shape, ordered, alongside the global
  ``vector`` / ``inference`` seed channels both transports carry;
* **append-order:** ``add_channel_columns`` appends new columns AFTER the
  originals, in declaration order;
* **tenant isolation / non-collision (#170):** a channel registered under tenant A
  is invisible to tenant B; B may register the same id with different columns
  without collision; A's channel is unchanged by B's; an unbound connection sees
  only the global seeds;
* **goldens:** the channel counts the emit froze (A's channel count, the
  annotated_by column count, the zero tenant leak, the zero collision) reproduce
  live.

NO consumer vocabulary: generic channel ids (``scored_by`` / ``annotated_by``),
opaque tenant UUIDs.
"""

from __future__ import annotations

import tempfile
import uuid

import pytest

jammi_ai = pytest.importorskip("jammi_ai")

from jammi_cookbook import contracts  # noqa: E402

_EVAL = contracts._dataset_dir("eval")
_HAVE_GOLDEN = (_EVAL / "golden_metrics.json").exists()


def _fresh_db():
    return jammi_ai.connect(f"file://{tempfile.mkdtemp(prefix='jammi_ch14_chan_')}")


def _fresh_tenant() -> str:
    return str(uuid.uuid4())


def test_register_then_list_round_trip():
    """A registered channel reappears in list_channels with its exact declared
    shape, alongside the global seed channels, ordered by (priority, channel_id)."""
    db = _fresh_db()
    tenant = _fresh_tenant()
    with db.tenant_scope(tenant):
        db.register_channel(
            "scored_by", priority=50, columns=[("score", "Float64"), ("model", "Utf8")]
        )
        listed = db.list_channels()

    scored = next(c for c in listed if c["channel_id"] == "scored_by")
    assert scored == {
        "channel_id": "scored_by",
        "priority": 50,
        "columns": [
            {"name": "score", "data_type": "Float64"},
            {"name": "model", "data_type": "Utf8"},
        ],
    }
    names = {c["channel_id"] for c in listed}
    assert {"vector", "inference"} <= names, "the global seed channels must be present"

    # The listing is ordered by (priority, channel_id) — a stable contract.
    keys = [(c["priority"], c["channel_id"]) for c in listed]
    assert keys == sorted(keys), "list_channels must be ordered by (priority, channel_id)"


def test_add_columns_appends_in_declaration_order():
    """add_channel_columns appends new columns after the originals, in order."""
    db = _fresh_db()
    tenant = _fresh_tenant()
    with db.tenant_scope(tenant):
        db.register_channel("annotated_by", priority=10, columns=[("label", "Utf8")])
        db.add_channel_columns(
            "annotated_by", columns=[("confidence", "Float32"), ("rank", "Int32")]
        )
        annotated = next(
            c for c in db.list_channels() if c["channel_id"] == "annotated_by"
        )
    assert annotated["columns"] == [
        {"name": "label", "data_type": "Utf8"},
        {"name": "confidence", "data_type": "Float32"},
        {"name": "rank", "data_type": "Int32"},
    ]


def test_redeclare_column_different_dtype_rejected():
    """Redeclaring an existing column with a DIFFERENT dtype raises the
    append-only violation — the engine rejects, it does not silently no-op."""
    db = _fresh_db()
    tenant = _fresh_tenant()
    with db.tenant_scope(tenant):
        db.register_channel("typed_chan", priority=1, columns=[("v", "Float32")])
        with pytest.raises(Exception) as err:
            db.add_channel_columns("typed_chan", columns=[("v", "Float64")])
    assert "cannot redeclare" in str(err.value)


def test_tenant_isolation_and_non_collision():
    """The #170 property, embedded-live: A's channel is invisible to B; B may
    register the same id with different columns without collision; A's channel is
    unchanged by B's; an unbound connection sees only the global seeds."""
    db = _fresh_db()
    tenant_a = _fresh_tenant()
    tenant_b = _fresh_tenant()

    with db.tenant_scope(tenant_a):
        db.register_channel("x_chan", priority=7, columns=[("w", "Float32")])

    # B does not see A's x_chan.
    with db.tenant_scope(tenant_b):
        b_before = {c["channel_id"] for c in db.list_channels()}
    assert "x_chan" not in b_before

    # B registers its OWN x_chan with different columns — no collision.
    with db.tenant_scope(tenant_b):
        db.register_channel("x_chan", priority=3, columns=[("note", "Utf8")])
        b_x = next(c for c in db.list_channels() if c["channel_id"] == "x_chan")
    assert b_x == {
        "channel_id": "x_chan",
        "priority": 3,
        "columns": [{"name": "note", "data_type": "Utf8"}],
    }

    # A's x_chan is unchanged by B's registration.
    with db.tenant_scope(tenant_a):
        a_x = next(c for c in db.list_channels() if c["channel_id"] == "x_chan")
    assert a_x == {
        "channel_id": "x_chan",
        "priority": 7,
        "columns": [{"name": "w", "data_type": "Float32"}],
    }

    # An unbound connection sees only the global (NULL-tenant) seeds.
    unbound = {c["channel_id"] for c in db.list_channels()}
    assert "x_chan" not in unbound
    assert {"vector", "inference"} <= unbound


@pytest.mark.skipif(not _HAVE_GOLDEN, reason="eval cache not emitted")
def test_channel_goldens_reproduce_live():
    """The channel counts the emit froze reproduce live on the embedded engine:
    A's channel count, the annotated_by column count, zero tenant leak, zero
    collision (#170)."""
    db = _fresh_db()
    tenant_a = _fresh_tenant()
    tenant_b = _fresh_tenant()

    with db.tenant_scope(tenant_a):
        db.register_channel(
            "scored_by", priority=50, columns=[("score", "Float64"), ("model", "Utf8")]
        )
        db.register_channel("annotated_by", priority=10, columns=[("label", "Utf8")])
        db.add_channel_columns(
            "annotated_by", columns=[("confidence", "Float32"), ("rank", "Int32")]
        )
        list_a = db.list_channels()

    a_names = {c["channel_id"] for c in list_a}
    contracts.assert_close("eval.channel.a_channel_count", float(len(a_names)))

    annotated = next(c for c in list_a if c["channel_id"] == "annotated_by")
    contracts.assert_close(
        "eval.channel.annotated_column_count", float(len(annotated["columns"]))
    )

    # tenant leak: B sees none of A's named channels.
    with db.tenant_scope(tenant_b):
        b_before = {c["channel_id"] for c in db.list_channels()}
    leak = len({"scored_by", "annotated_by"} & b_before)
    contracts.assert_close("eval.channel.tenant_leak", float(leak))

    # collision: B's scored_by (different columns) does not become A's.
    with db.tenant_scope(tenant_b):
        db.register_channel("scored_by", priority=3, columns=[("note", "Utf8")])
        b_scored = next(c for c in db.list_channels() if c["channel_id"] == "scored_by")
    with db.tenant_scope(tenant_a):
        a_scored = next(c for c in db.list_channels() if c["channel_id"] == "scored_by")
    collision = 1.0 if a_scored["columns"] == b_scored["columns"] else 0.0
    contracts.assert_close("eval.channel.collision", collision)
