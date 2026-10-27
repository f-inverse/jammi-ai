"""A source is removed with `remove_source`, on both transports, by its owner only.

The owner's `remove_source` takes the source out of `list_sources` and
`describe_source`, a query over it then fails, and the name can be registered
again. Another tenant's attempt on it raises `NotFound` — the class
`describe_source` answers that tenant with — and leaves the source in place; so
does removing a name never registered, or one already removed. The embedded
engine and a real `jammi-server` give the same answers.

Selected by the `live_server` and `embedded` markers: it needs a built
`jammi-server` (`JAMMI_SERVER_BIN`) and the in-process engine as the parity peer.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import jammi
from jammi.errors import NotFound

pytestmark = [pytest.mark.live_server, pytest.mark.embedded]

TENANT_A = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a"
TENANT_B = "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9b"


def _refused(call) -> str:
    """The class `call` raises, which must be `NotFound`."""
    with pytest.raises(NotFound) as refused:
        call()
    return type(refused.value).__name__


def _observe(db, url: str) -> dict:
    """What each step of the issue's reproduction answers on `db`."""
    seen = {}
    with db.tenant_scope(TENANT_A):
        db.add_source("own", url=url, format="parquet")
    with db.tenant_scope(TENANT_B):
        seen["b_removes_a_source"] = _refused(lambda: db.remove_source("own"))
        seen["b_removes_a_name_never_registered"] = _refused(lambda: db.remove_source("never"))
    with db.tenant_scope(TENANT_A):
        seen["a_rows_after_b"] = db.sql("SELECT COUNT(*) AS n FROM own.public.orders").to_pylist()
        db.remove_source("own")
        seen["a_lists_after_remove"] = [s["source_id"] for s in db.list_sources()]
        seen["a_describes_after_remove"] = db.describe_source("own")
        seen["a_queries_after_remove"] = _refused(
            lambda: db.sql("SELECT COUNT(*) AS n FROM own.public.orders")
        )
        seen["a_removes_again"] = _refused(lambda: db.remove_source("own"))
        db.add_source("own", url=url, format="parquet")
        seen["a_rows_after_registering_again"] = db.sql(
            "SELECT COUNT(*) AS n FROM own.public.orders"
        ).to_pylist()
    return seen


def test_remove_source_removes_only_the_callers_own_source_on_both_transports(
    tmp_path, live_server_on
):
    orders = tmp_path / "orders.parquet"
    pq.write_table(pa.table({"id": [1, 2, 3], "total": [9.5, 12.0, 3.25]}), orders)
    url = f"file://{orders}"

    embedded_dir = tmp_path / "embedded"
    embedded_dir.mkdir()
    db = jammi.connect(f"file://{embedded_dir}")
    try:
        assert callable(db.remove_source)
        embedded = _observe(db, url)
    finally:
        db.close()

    served_dir = tmp_path / "served"
    served_dir.mkdir()
    with live_server_on(served_dir) as endpoint:
        remote = jammi.connect(endpoint)
        try:
            assert callable(remote.remove_source)
            served = _observe(remote, url)
        finally:
            remote.close()

    assert embedded == {
        "b_removes_a_source": "NotFound",
        "b_removes_a_name_never_registered": "NotFound",
        "a_rows_after_b": [{"n": 3}],
        "a_lists_after_remove": [],
        "a_describes_after_remove": None,
        "a_queries_after_remove": "NotFound",
        "a_removes_again": "NotFound",
        "a_rows_after_registering_again": [{"n": 3}],
    }
    assert served == embedded
