"""Hermetic unit tests for the loaders (K2) — no network.

The full download+register path is exercised by the executable ``datasets``
chapter in CI (checksum-gated, cache-backed). These tests cover the pure logic:
the checksum gate, the deterministic connected-subset selection, and the schemas.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import datasets
from jammi_cookbook.datasets import _ArxivRaw


def test_download_checksum_gate(tmp_path):
    import hashlib

    dest = tmp_path / "blob.bin"
    payload = b"pinned-content"
    dest.write_bytes(payload)  # pre-existing → _download skips the network fetch
    good = hashlib.sha256(payload).hexdigest()

    # matching digest returns the path
    assert datasets._download("https://unused", good, dest=dest) == dest
    # a mismatch fails loudly and removes the tampered file
    with pytest.raises(ValueError, match="checksum mismatch"):
        datasets._download("https://unused", "0" * 64, dest=dest)
    assert not dest.exists()


def _line_graph(n: int) -> _ArxivRaw:
    """A path graph 0-1-2-…-(n-1): trivially connected, every node degree ≤2."""
    edges = [(i, i + 1) for i in range(n - 1)]
    return _ArxivRaw(
        num_nodes=n,
        edges=edges,
        labels=[i % 3 for i in range(n)],
        years=[2018 + (i % 3) for i in range(n)],
        node2pid=[1000 + i for i in range(n)],
        label_names=["a", "b", "c"],
        split={"train": [0], "valid": [1], "test": list(range(2, n))},
    )


def test_connected_subset_is_deterministic_and_sized():
    raw = _line_graph(50)
    a = datasets._connected_subset(raw, 10)
    b = datasets._connected_subset(raw, 10)
    assert a == b  # pure function of the graph
    assert len(a) == 10


def test_connected_subset_is_connected():
    # Two disjoint components; the subset must stay within one (BFS from a hub).
    edges = [(0, 1), (1, 2), (2, 0)] + [(10, 11), (11, 12)]
    raw = _ArxivRaw(
        num_nodes=13,
        edges=edges,
        labels=[0] * 13,
        years=[2019] * 13,
        node2pid=list(range(13)),
        label_names=["x"],
        split={"train": [], "valid": [], "test": []},
    )
    sub = set(datasets._connected_subset(raw, 3))
    # The densest component is the triangle {0,1,2}; the subset is exactly it.
    assert sub == {0, 1, 2}


def test_subset_capped_when_component_smaller_than_size():
    edges = [(0, 1), (1, 2), (2, 0)]
    raw = _ArxivRaw(
        num_nodes=3,
        edges=edges,
        labels=[0, 1, 2],
        years=[2019] * 3,
        node2pid=[7, 8, 9],
        label_names=["x", "y", "z"],
        split={"train": [], "valid": [], "test": []},
    )
    assert len(datasets._connected_subset(raw, 100)) == 3


def test_schemas_have_expected_columns():
    assert datasets._paper_schema().names == ["paper_id", "title", "abstract", "subject", "year"]
    assert datasets._airport_schema().names == datasets._AIRPORT_COLUMNS
