"""Lexical (BM25) retrieval: `build_lexical_index` materialises a source's text
and `lexical_search` ranks it — the request each transport sends, the rows the
embedded engine returns, and the same table from a live server over the same
catalog.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import jammi
from jammi._assembly import build_lexical_index_request, build_lexical_search_request
from jammi._generated.jammi.v1 import pipeline_pb2

REPO = Path(__file__).resolve().parents[3]
PATENTS_URL = f"file://{REPO / 'tests' / 'fixtures' / 'patents.parquet'}"


def test_the_build_request_names_the_analyzer_and_refuses_an_unknown_one() -> None:
    request = build_lexical_index_request("patents", columns=["title", "abstract"], key="id")
    assert list(request.columns) == ["title", "abstract"] and request.key_column == "id"
    assert request.analyzer == pipeline_pb2.LexicalAnalyzer.LEXICAL_ANALYZER_ENGLISH
    raw = build_lexical_index_request("patents", columns=["title"], key="id", analyzer="raw")
    assert raw.analyzer == pipeline_pb2.LexicalAnalyzer.LEXICAL_ANALYZER_RAW
    with pytest.raises(ValueError, match="analyzer"):
        build_lexical_index_request("patents", columns=["title"], key="id", analyzer="french")


def test_the_search_request_carries_only_what_was_given() -> None:
    bare = build_lexical_search_request("patents", text="quantum", k=5)
    assert not bare.HasField("filter") and not bare.HasField("lexical_table")
    named = build_lexical_search_request(
        "patents", text="quantum", k=5, filter="year > 2020", lexical_table="t"
    )
    assert (named.filter, named.lexical_table) == ("year > 2020", "t")


def _searches(db):
    return (
        db.lexical_search("patents", text="quantum computing", k=5),
        db.lexical_search("patents", text="quantum", k=2, filter="year >= 2022",
                          select=["id", "year"]),
    )


@pytest.mark.embedded
def test_a_lexical_search_ranks_and_hydrates_the_matching_patents(tmp_path) -> None:
    db = jammi.connect(f"file://{tmp_path}")
    try:
        db.add_source("patents", url=PATENTS_URL, format="parquet")
        table = db.build_lexical_index("patents", columns=["title", "abstract"], key="id")
        ranked, filtered = _searches(db)
        named = db.lexical_search("patents", text="quantum", k=20, lexical_table=table)
    finally:
        db.close()

    assert ranked.column("bm25_rank").to_pylist() == list(range(ranked.num_rows))
    scores = ranked.column("bm25_score").to_pylist()
    assert scores == sorted(scores, reverse=True)
    assert all(by == ["bm25"] for by in ranked.column("retrieved_by").to_pylist())
    assert all("uantum" in t for t in ranked.column("title").to_pylist()[:2])
    assert filtered.num_rows == 2 and all(y >= 2022 for y in filtered.column("year").to_pylist())
    # Every patent whose title or abstract says "quantum", and no other.
    assert sorted(named.column("id").to_pylist()) == [1, 4, 7, 11, 12, 16]


@pytest.mark.live_server
@pytest.mark.embedded
def test_remote_and_embedded_lexical_search_agree(tmp_path, live_server_on) -> None:
    artifact_dir = tmp_path / "engine"
    artifact_dir.mkdir()
    db = jammi.connect(f"file://{artifact_dir}")
    try:
        db.add_source("patents", url=PATENTS_URL, format="parquet")
        db.build_lexical_index("patents", columns=["title", "abstract"], key="id")
        embedded = _searches(db)
    finally:
        db.close()

    with live_server_on(artifact_dir) as endpoint:
        remote = jammi.connect(endpoint)
        try:
            for local, over_the_wire in zip(embedded, _searches(remote)):
                assert over_the_wire.schema == local.schema
                assert over_the_wire.to_pylist() == local.to_pylist()
        finally:
            remote.close()
