"""Hermetic unit tests for the dataset loaders — no network.

The chapters run the register path at both scales. These cover the pure logic
the `full` scale's tables are built with — the checksum gate, the fetch retry,
the breadth-first ball and its best-connected core, the time split — and that
the committed `small` fixture is the core of the ball it names.
"""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from jammi_cookbook import datasets, fixtures

PAYLOAD = b"pinned-content"
GOOD = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "_CACHE", tmp_path)
    return tmp_path


def test_a_cached_download_is_checked_and_a_mismatch_is_removed(cache):
    dest = cache / "raw" / "blob.bin"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(PAYLOAD)  # already cached: no fetch
    assert datasets._download("https://unused", GOOD, name="blob.bin") == dest
    with pytest.raises(ValueError, match="checksum mismatch"):
        datasets._download("https://unused", "0" * 64, name="blob.bin")
    assert not dest.exists()


def test_a_source_that_fails_before_it_serves_is_retried(cache, monkeypatch):
    served = []

    class FlakyThenServes(BaseHTTPRequestHandler):
        def do_GET(self):
            served.append(self.path)
            if len(served) < 3:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), FlakyThenServes)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(datasets._RETRY, "backoff_factor", 0.0)
    try:
        url = f"http://127.0.0.1:{server.server_port}/blob.bin"
        assert datasets._download(url, GOOD, name="blob.bin").read_bytes() == PAYLOAD
        assert len(served) == 3, "two 503s retried, the third request served"
    finally:
        server.shutdown()


def test_the_ball_grows_breadth_first_from_the_highest_degree_node():
    # A hub (2) with three spokes, one of which (3) leads on to a tail (4-5).
    edges = [(2, 0), (2, 1), (2, 3), (3, 4), (4, 5)]
    assert datasets._ball(6, edges, 4) == [2, 0, 1, 3]
    assert datasets._ball(6, edges, 6) == [2, 0, 1, 3, 4, 5]
    assert datasets._ball(6, edges, 100) == [2, 0, 1, 3, 4, 5]


def test_every_prefix_of_a_larger_ball_is_the_smaller_ball():
    edges = [(i, (i * 7 + 3) % 40) for i in range(40)] + [(i, i + 1) for i in range(39)]
    big = datasets._ball(40, edges, 30)
    assert all(datasets._ball(40, edges, n) == big[:n] for n in range(1, 30))


def test_the_core_keeps_the_best_connected_nodes_in_ball_order():
    ball = [2, 0, 1, 3, 4, 5]
    edges = [(2, 0), (2, 1), (2, 3), (3, 4), (4, 5), (0, 1), (9, 2)]
    # Inside-ball degrees: 2→3, 0→2, 1→2, 3→2, 4→2, 5→1; (9, 2) is outside.
    assert datasets._core(ball, edges, 4) == [2, 0, 1, 3]


def test_the_time_split_is_by_year():
    papers = pa.table({"paper_id": ["a", "b", "c", "d"], "year": [2016, 2017, 2018, 2019]})
    assert datasets.time_split(papers) == {"train": ["a", "b"], "valid": ["c"], "test": ["d"]}


def test_the_small_fixture_is_the_core_and_keeps_only_its_citations():
    papers = pq.read_table(fixtures.path("arxiv_small/arxiv_papers.parquet"))
    cites = pq.read_table(fixtures.path("arxiv_small/arxiv_cites.parquet"))
    ids = set(papers.column("paper_id").to_pylist())
    assert papers.num_rows == datasets.ARXIV_CORE == len(ids)
    assert papers.column_names == ["paper_id", "title", "abstract", "subject", "year"]
    assert set(cites.column("src").to_pylist()) | set(cites.column("dst").to_pylist()) <= ids
    split = datasets.time_split(papers)
    assert all(split.values()), "every era is represented at small scale"
