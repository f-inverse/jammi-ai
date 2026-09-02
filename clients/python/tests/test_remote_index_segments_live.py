"""`list_index_segments` reads IDENTICALLY on both transports, against one catalog.

Stands up a real CPU `jammi-server` over an artifact directory an EMBEDDED
engine seeded and released first, and compares the remote `RemoteDatabase`
listing against the embedded `EmbeddedBackend` listing of the very same
`index_segments` rows. Comparing two arms' reads of ONE catalog is the parity
claim; two separately-built catalogs would only prove each arm self-consistent.

The divergence-prone input is the MULTI-segment table whose segments were
inserted out of `segment_id` order: a one-segment table would pass even if the
ordering or the repeated-field decode diverged. The empty cases (an unknown
table) ride the same comparison.

Ordering is forced by the single-process SQLite contract and is the point, not
an inconvenience: the embedded seeder must have RELEASED the catalog (`close()`)
before the server opens it, and the embedded read is taken before the server
starts. The Rust peer of this test (`crates/jammi-server/tests/it/
grpc_remote_list.rs`) compares the same two answers as serialized wire bytes;
this one compares the two PYTHON dict lists a caller actually holds.

Skipped unless `JAMMI_SERVER_BIN` points at a built `jammi-server` AND the
`[embedded]` extra is installed (the seeder needs the in-process engine) — the
same gate shape every other live module in this directory declares.
"""

from __future__ import annotations

import os
import sqlite3
from importlib.util import find_spec
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import jammi

SERVER_BIN = os.environ.get("JAMMI_SERVER_BIN")

pytestmark = pytest.mark.skipif(
    not (SERVER_BIN and Path(SERVER_BIN).is_file()) or find_spec("jammi_native") is None,
    reason="needs JAMMI_SERVER_BIN and the [embedded] extra (the seeder is the in-process engine)",
)

DIM = 4
N_CORPUS = 6
TABLE_SOURCE = "segparity_vectors"


def _write_corpus(dir_path: Path) -> str:
    ids = [f"row-{i}" for i in range(N_CORPUS)]
    vectors = [[float(i + j) for j in range(DIM)] for i in range(N_CORPUS)]
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


def _seed(artifact_dir: Path, corpus_url: str) -> tuple[str, list]:
    """Build a MULTI-segment table through the embedded engine, read it, and
    RELEASE the catalog. Returns `(table_name, embedded_listing)`.

    Segment 0 is the engine's own (written by `import_embeddings`). Segments 2
    and 1 are appended in that order through a raw `sqlite3` connection — no
    public verb appends a second segment to an existing table today — under the
    close-before-inject discipline: the engine has released the file before the
    raw write, and the raw connection is closed before the engine reopens.
    """
    db = jammi.connect(f"file://{artifact_dir}")
    db.add_source(TABLE_SOURCE, url=corpus_url, format="parquet")
    table = db.import_embeddings(
        source=TABLE_SOURCE,
        model="synthetic-segment-vectors",
        vectors_url=corpus_url,
        key="_row_id",
        dimensions=DIM,
    )
    db.close()

    conn = sqlite3.connect(str(artifact_dir / "catalog.db"))
    try:
        conn.executemany(
            "INSERT INTO index_segments (table_name, segment_id, index_path, "
            "row_count, tenant_id) VALUES (?, ?, ?, ?, NULL)",
            [
                (table, 2, f"file:///idx/{table}-2", 11),
                (table, 1, f"file:///idx/{table}-1", 7),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    db = jammi.connect(f"file://{artifact_dir}")
    try:
        embedded = db.list_index_segments(table)
        embedded_unknown = db.list_index_segments("no_such_table_anywhere")
    finally:
        db.close()

    assert [s["segment_id"] for s in embedded] == [0, 1, 2], embedded
    assert embedded_unknown == []
    return table, embedded


def test_remote_and_embedded_list_index_segments_agree(tmp_path, live_server_on):
    """The remote listing equals the embedded listing, entry for entry, key for
    key — on a three-segment table whose rows were inserted out of order, and on
    the unknown-table empty case.

    Every dict is compared whole, not spot-checked: a `row_count` that lost its
    `uint64` width, an `index_path` re-shaped in either mapper, or an ordering
    that came back in insertion order on one arm only, all fail here.
    """
    artifact_dir = tmp_path / "engine"
    artifact_dir.mkdir()
    corpus = _write_corpus(tmp_path)
    table, embedded = _seed(artifact_dir, corpus)

    with live_server_on(artifact_dir) as endpoint:
        remote = jammi.connect(endpoint)
        try:
            assert remote.list_index_segments(table) == embedded
            assert remote.list_index_segments("no_such_table_anywhere") == []
        finally:
            remote.close()

    # And the values themselves are the rows the catalog holds, so the equality
    # above is agreement on real content rather than on two empty lists.
    assert embedded[1] == {
        "segment_id": 1,
        "index_path": f"file:///idx/{table}-1",
        "row_count": 7,
    }
    assert embedded[2] == {
        "segment_id": 2,
        "index_path": f"file:///idx/{table}-2",
        "row_count": 11,
    }
    assert embedded[0]["segment_id"] == 0
    assert embedded[0]["row_count"] == N_CORPUS
