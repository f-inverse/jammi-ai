"""`dimensions` serves a Matryoshka prefix: `generate_embeddings` stores the
model's leading coordinates, renormalised, `encode_query` encodes to the same
width, a search at that width finds the rows, and a width the model does not
have is refused — on both transports.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import jammi
from jammi._assembly import build_encode_query_request, build_generate_embeddings_request
from jammi.errors import InvalidArgument

REPO = Path(__file__).resolve().parents[3]
PATENTS_URL = f"file://{REPO / 'tests' / 'fixtures' / 'patents.parquet'}"
TINY_BERT = f"local:{REPO / 'cookbook' / 'fixtures' / 'tiny_bert'}"


def _prefix(vector, width):
    head = vector[:width]
    norm = math.sqrt(sum(v * v for v in head))
    return [v / norm for v in head]


def test_the_requests_carry_a_width_only_when_one_is_asked_for() -> None:
    full = build_generate_embeddings_request(source="s", model="m", columns=["t"], key="id")
    assert not full.HasField("dimensions")
    narrow = build_generate_embeddings_request(
        source="s", model="m", columns=["t"], key="id", dimensions=8
    )
    assert narrow.dimensions == 8
    assert build_encode_query_request(model="m", query="q", dimensions=8).dimensions == 8
    assert not build_encode_query_request(model="m", query="q").HasField("dimensions")


@pytest.mark.embedded
def test_a_prefix_table_serves_and_searches_at_its_width(tmp_path) -> None:
    db = jammi.connect(f"file://{tmp_path}")
    try:
        db.add_source("patents", url=PATENTS_URL, format="parquet")
        table = db.generate_embeddings(
            source="patents", model=TINY_BERT, columns=["abstract"], key="id", dimensions=8
        )
        full = db.encode_query(model=TINY_BERT, query="quantum error correction")
        narrow = db.encode_query(model=TINY_BERT, query="quantum error correction", dimensions=8)
        hits = db.search("patents", query=narrow, k=3, embedding_table=table, exact=True)
        described = db.describe_table(table)
        with pytest.raises(InvalidArgument, match="cannot serve 64 dimensions"):
            db.generate_embeddings(
                source="patents", model=TINY_BERT, columns=["abstract"], key="id", dimensions=64
            )
    finally:
        db.close()

    assert len(full) == 32 and len(narrow) == 8
    assert all(abs(a - b) < 1e-5 for a, b in zip(narrow, _prefix(full, 8)))
    assert hits.num_rows == 3
    assert described["descriptor"]["dimensions"] == 8


@pytest.mark.live_server
@pytest.mark.embedded
def test_remote_and_embedded_encode_the_same_prefix(tmp_path, live_server_on) -> None:
    db = jammi.connect(f"file://{tmp_path / 'engine'}")
    try:
        embedded = db.encode_query(model=TINY_BERT, query="a battery electrode", dimensions=8)
    finally:
        db.close()
    with live_server_on(tmp_path / "engine") as endpoint:
        remote = jammi.connect(endpoint)
        try:
            over_the_wire = remote.encode_query(
                model=TINY_BERT, query="a battery electrode", dimensions=8
            )
        finally:
            remote.close()
    assert over_the_wire == pytest.approx(embedded, abs=1e-6)
