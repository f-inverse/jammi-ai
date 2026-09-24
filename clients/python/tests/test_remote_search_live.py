"""`search` returns the same table on both transports, against one catalog.

An EMBEDDED engine embeds a corpus and releases the catalog; a real CPU
`jammi-server` then serves the same artifact directory. The same query is run
through the embedded `EmbeddedBackend` and the remote `RemoteDatabase`, and
the two `pyarrow.Table`s are compared whole — schema and rows: the hydrated
source columns, the retrieval provenance and the similarity. A `select`
projection is compared the same way, so a caller swaps transports without
changing what it reads off the result.

Selected by the `live_server` and `embedded` markers: it needs a built
`jammi-server` (`JAMMI_SERVER_BIN`) and the in-process engine as the parity peer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import jammi

pytestmark = [pytest.mark.live_server, pytest.mark.embedded]

REPO = Path(__file__).resolve().parents[3]
PATENTS_URL = f"file://{REPO / 'tests' / 'fixtures' / 'patents.parquet'}"
TINY_BERT = f"local:{REPO / 'cookbook' / 'fixtures' / 'tiny_bert'}"


def _searches(db, query):
    return (
        db.search("patents", query=query, k=5),
        db.search("patents", query=query, k=5, select=["_row_id", "title", "similarity"]),
        db.search("patents", query=query, k=5, filter="year >= 2020"),
    )


def test_remote_and_embedded_search_agree(tmp_path, live_server_on):
    artifact_dir = tmp_path / "engine"
    artifact_dir.mkdir()

    db = jammi.connect(f"file://{artifact_dir}")
    try:
        db.add_source("patents", url=PATENTS_URL, format="parquet")
        db.generate_embeddings(source="patents", model=TINY_BERT, columns=["abstract"], key="id")
        query = db.encode_query(model=TINY_BERT, query="a battery electrode")
        embedded = _searches(db, query)
    finally:
        db.close()

    full, projected, filtered = embedded
    assert full.num_rows == 5 and "abstract" in full.column_names
    assert "title" in projected.column_names and "abstract" not in projected.column_names

    with live_server_on(artifact_dir) as endpoint:
        remote = jammi.connect(endpoint)
        try:
            for local, over_the_wire in zip(embedded, _searches(remote, query)):
                assert over_the_wire.schema == local.schema
                assert over_the_wire.to_pylist() == local.to_pylist()
        finally:
            remote.close()
