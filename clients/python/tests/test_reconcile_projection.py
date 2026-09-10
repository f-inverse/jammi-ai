"""Hermetic projection oracle for `_reconcile_report_to_dict`.

`Database.reconcile`'s remote arm must reproduce the embed wheel's dict shape
EXACTLY — the embedded `reconcile` returns the engine's `ReconcileReport`
struct through `serde_json` (`serializable_to_pydict`), so the projection
contract is that serde shape: all 14 fields
(`scope`/`applied`/`rows_failed`/`rows_failed_count`/`orphans`/
`orphan_count`/`pending`/`pending_count`/`unattributed`/
`unattributed_count`/`damaged`/`damaged_count`/`truncated`/
`bytes_reclaimed`), the same key names, spelled the same way.

The proto message below is hand-constructed with literal values (not built
from any fixture the projection itself could have produced), every list
non-empty, `truncated=True`, a `damaged` entry present, and every `*_count`
deliberately different from its corresponding list's length — so a
projection that silently derived a count from `len(list)` instead of
projecting the wire field would be caught here too.

Hermetic: no server, no embedded engine — only a constructed proto.
"""

from __future__ import annotations

from jammi._database import _reconcile_report_to_dict
from jammi._generated.jammi.v1 import catalog_pb2

_RECONCILE_REPORT_DICT_KEYS = {
    "scope",
    "applied",
    "rows_failed",
    "rows_failed_count",
    "orphans",
    "orphan_count",
    "pending",
    "pending_count",
    "unattributed",
    "unattributed_count",
    "damaged",
    "damaged_count",
    "truncated",
    "bytes_reclaimed",
}


def _populated_report() -> catalog_pb2.ReconcileReport:
    return catalog_pb2.ReconcileReport(
        scope="tenant:22222222-2222-4222-8222-222222222222",
        applied=True,
        rows_failed=["ra", "rb"],
        rows_failed_count=41,
        orphans=["oa"],
        orphan_count=17,
        pending=["pa", "pb", "pc"],
        pending_count=29,
        unattributed=["ua"],
        unattributed_count=8,
        damaged=["da"],
        damaged_count=5,
        truncated=True,
        bytes_reclaimed=99999,
    )


def test_reconcile_report_to_dict_projects_all_fourteen_fields():
    """The whole row and nothing else: key set matches the engine's full
    `ReconcileReport` shape."""
    projected = _reconcile_report_to_dict(_populated_report())
    assert set(projected) == _RECONCILE_REPORT_DICT_KEYS, (
        f"_reconcile_report_to_dict returned {set(projected)} != "
        f"{_RECONCILE_REPORT_DICT_KEYS}"
    )


def test_reconcile_report_to_dict_round_trips_every_value():
    """Every value survives the projection unchanged — including a
    `*_count` that deliberately disagrees with its list's length, proving
    the count is projected from the wire field, never recomputed from
    `len(list)`."""
    report = _populated_report()
    projected = _reconcile_report_to_dict(report)
    assert projected == {
        "scope": "tenant:22222222-2222-4222-8222-222222222222",
        "applied": True,
        "rows_failed": ["ra", "rb"],
        "rows_failed_count": 41,
        "orphans": ["oa"],
        "orphan_count": 17,
        "pending": ["pa", "pb", "pc"],
        "pending_count": 29,
        "unattributed": ["ua"],
        "unattributed_count": 8,
        "damaged": ["da"],
        "damaged_count": 5,
        "truncated": True,
        "bytes_reclaimed": 99999,
    }
