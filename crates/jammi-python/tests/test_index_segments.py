"""`list_index_segments` — the per-table ANN segment listing on the Python surface.

A result table's ANN index is a SET of immutable segments (migration 025), one
`index_segments` catalog row each. Before this verb those rows were reachable
through no public surface at all: `db.sql()` federates result tables and
external sources, not the catalog's own tables, so a caller who needed them had
to close the engine and open its SQLite file directly — the out-of-contract
topology the catalog-and-broker guide documents as unarbitrable.

This module pins what the PYTHON surface owes: the embedded arm's dict shape,
its `segment_id` ordering on the divergence-prone MULTI-segment input, and the
four-way empty contract (no segments / flat index / unknown table / a table this
tenant cannot resolve — all the same empty list, never an error, so the verb is
not an existence oracle). The cross-transport byte-parity of the same reads
lives in `clients/python/tests/test_remote_index_segments_live.py`, which needs
a real `jammi-server`; the Rust `grpc_remote_list.rs` peer compares the same
answers as serialized wire bytes.

**Seeding a multi-segment table.** Only the engine's own append path mints
segments, and no public verb appends a second one to an existing table today, so
the extra rows are written through a raw `sqlite3` connection under the SAME
close-before-inject discipline the two other sanctioned raw touch points in this
suite use (`test_embedded_training.py`, `test_conformance.py`): the engine is
`close()`d — the bounded catalog release — before any raw write, and never
reopened until that connection is closed. The parent `result_tables` row is
always created by the ENGINE (`import_embeddings`), never hand-built, so the FK
parent the tenant gate resolves through is a real row.

Hermetic: a synthetic parquet corpus, `import_embeddings` (no encoder, no GPU),
into a temp artifact dir. No network.
"""

from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import jammi

DIM = 8
N_CORPUS = 12
SEED = 446

# The tenant ids are opaque UUIDs — the engine requires a UUID, and a test that
# spelled a consumer into one would be naming a consumer in the surface.
TENANT_A = "11111111-1111-4111-8111-111111111111"
TENANT_B = "22222222-2222-4222-8222-222222222222"


def _write_corpus(dir_path: Path) -> str:
    rng = random.Random(SEED)
    ids = [f"row-{i}" for i in range(N_CORPUS)]
    vectors = [[rng.uniform(-1.0, 1.0) for _ in range(DIM)] for _ in ids]
    path = dir_path / "corpus.parquet"
    pq.write_table(
        pa.table(
            {
                "_row_id": ids,
                "vector": pa.array(vectors, type=pa.list_(pa.float32(), DIM)),
            }
        ),
        path,
    )
    return f"file://{path}"


def _register_table(db, corpus_url: str, *, source: str = "vectors") -> str:
    """A REAL, ready embedding table through the public surface only:
    `add_source` + `import_embeddings`. The engine builds its index and records
    segment 0, so the listing below reads a row the engine itself wrote."""
    db.add_source(source, url=corpus_url, format="parquet")
    return db.import_embeddings(
        source=source,
        model="synthetic-segment-vectors",
        vectors_url=corpus_url,
        key="_row_id",
        dimensions=DIM,
    )


def _insert_segments(catalog_db: Path, table: str, rows: List[tuple]) -> None:
    """Write extra `index_segments` rows through a raw `sqlite3` connection.

    Only ever called with NO engine connection attached to this catalog (see the
    module docstring): the engine holds a process-scoped exclusive lock through
    the `unix-excl` VFS and keeps its wal-index in heap memory, so a raw write
    overlapping a live engine connection is out of contract.
    """
    conn = sqlite3.connect(str(catalog_db))
    try:
        conn.executemany(
            "INSERT INTO index_segments (table_name, segment_id, index_path, "
            "row_count, tenant_id) VALUES (?, ?, ?, ?, ?)",
            [(table, seg, path, count, tenant) for seg, path, count, tenant in rows],
        )
        conn.commit()
    finally:
        conn.close()


def test_list_index_segments_returns_the_engines_own_segment_zero(tmp_path: Path) -> None:
    """A freshly imported embedding table lists exactly the segment the ENGINE
    recorded for it — one entry, `segment_id` 0, with the three projected keys
    and nothing else.

    This is the read side proven against a real producer: no row here was
    hand-written, so the dict is the catalog's own `index_segments` row as the
    engine minted it (the `index_path` is the segment's real sidecar-bundle base
    URL, and `row_count` is the corpus size the index actually holds).
    """
    corpus = _write_corpus(tmp_path)
    db = jammi.connect(f"file://{tmp_path / 'engine'}")
    try:
        table = _register_table(db, corpus)
        segments = db.list_index_segments(table)

        assert isinstance(segments, list)
        assert len(segments) == 1, segments
        entry = segments[0]
        assert set(entry.keys()) == {"segment_id", "index_path", "row_count"}
        assert entry["segment_id"] == 0
        assert isinstance(entry["index_path"], str) and entry["index_path"]
        assert entry["row_count"] == N_CORPUS
    finally:
        db.close()


def test_list_index_segments_orders_by_segment_id_not_insertion_order(
    tmp_path: Path,
) -> None:
    """The MULTI-segment case, with the segments inserted OUT of order: the
    listing comes back in `segment_id` order.

    A single-segment table would pass even if the ordering were whatever the
    catalog happened to return, which is exactly why the divergence-prone input
    is the one pinned. Segments 2 and 1 are appended after the engine's own 0,
    in that order, so insertion order and `segment_id` order disagree.
    """
    corpus = _write_corpus(tmp_path)
    artifact_dir = tmp_path / "engine"
    catalog_db = artifact_dir / "catalog.db"

    db = jammi.connect(f"file://{artifact_dir}")
    table = _register_table(db, corpus)
    engine_segment = db.list_index_segments(table)[0]
    # The release point — no engine connection is attached past this line.
    db.close()

    _insert_segments(
        catalog_db,
        table,
        [
            (2, f"file:///idx/{table}-2", 11, None),
            (1, f"file:///idx/{table}-1", 7, None),
        ],
    )

    db = jammi.connect(f"file://{artifact_dir}")
    try:
        segments = db.list_index_segments(table)
        assert [s["segment_id"] for s in segments] == [0, 1, 2], segments
        assert segments == [
            engine_segment,
            {"segment_id": 1, "index_path": f"file:///idx/{table}-1", "row_count": 7},
            {"segment_id": 2, "index_path": f"file:///idx/{table}-2", "row_count": 11},
        ]
    finally:
        db.close()


def test_unknown_table_and_a_peer_tenants_table_list_the_same_empty(
    tmp_path: Path,
) -> None:
    """The empty contract, on the two arms that must be indistinguishable.

    Tenant A owns a segmented table. Tenant B asks for it BY NAME and gets `[]`
    — byte-identical to what B gets for a name that exists nowhere. That
    identity is the point: an error, or any other difference between the two,
    would turn the verb into an existence oracle for a peer tenant's table
    names. A's own read still returns its segment, so the empty answer is a
    tenant predicate and not a broken listing.
    """
    corpus = _write_corpus(tmp_path)
    db = jammi.connect(f"file://{tmp_path / 'engine'}")
    try:
        db.set_tenant(TENANT_A)
        table = _register_table(db, corpus)
        a_segments = db.list_index_segments(table)
        assert len(a_segments) == 1, a_segments

        db.set_tenant(TENANT_B)
        peer = db.list_index_segments(table)
        unknown = db.list_index_segments("no_such_table_anywhere")
        assert peer == [] == unknown

        # And A still sees its own — the predicate is the tenant, not a
        # listing that stopped working.
        db.set_tenant(TENANT_A)
        assert db.list_index_segments(table) == a_segments
    finally:
        db.close()


def test_list_index_segments_raises_the_typed_error_after_close(tmp_path: Path) -> None:
    """The FFI-boundary guard covers the new verb too: after `close()` it raises
    the typed `BackendError` every other verb raises, never a silent `[]` that
    would read as "this table has no segments"."""
    from jammi.errors import BackendError

    db = jammi.connect(f"file://{tmp_path}")
    assert db.list_index_segments("anything") == []
    db.close()
    with pytest.raises(BackendError, match="closed"):
        db.list_index_segments("anything")
