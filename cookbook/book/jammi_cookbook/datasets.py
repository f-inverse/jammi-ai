"""The book's two graph datasets, at either scale — Air Routes and ogbn-arxiv.

* **Air Routes** (Neptune's own teaching dataset; permissive, from
  ``krlawrence/graph``) — 3504 airports, the airport↔airport ``route`` graph
  and the continent→country→airport ``contains`` hierarchy. Small enough to run
  whole, so both scales read the committed ``air_routes`` fixture; the scales
  differ only in the encoder.
* **ogbn-arxiv** (ODC-BY; Open Graph Benchmark) — ~169k CS papers, ~1.17M
  citation edges, 40 subject classes, title + abstract. ``full`` runs a
  connected ball of the citation graph — 4,000 papers collected breadth-first
  from the highest-degree paper, downloaded from the pinned archive. ``small``
  runs that ball's 400 best-connected papers (most citations inside the ball),
  committed as the ``arxiv_small`` fixture: one dataset at two sizes, the
  small one keeping the full one's citation density and subject homophily.

The ogbn-arxiv time split (train ≤ 2017, valid 2018, test ≥ 2019) is a
property of each paper's ``year``, so it is derived, never stored.

Every download is checksum-gated against a pinned digest: a changed source
fails loudly rather than drifting the book's numbers. Author-time,
``python -m jammi_cookbook.datasets`` rewrites the committed fixtures from the
pinned sources.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import os
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import fixtures
from .scale import Scale

# The full scale's breadth-first ball, and the size of its best-connected core
# the small scale runs.
ARXIV_BALL = 4000
ARXIV_CORE = 400

# The time split, by publication year.
TRAIN_UNTIL, VALID_YEAR = 2017, 2018

# Where downloads and the tables built from them are kept.
_CACHE = Path(os.environ.get("JAMMI_COOKBOOK_CACHE", Path.home() / ".cache" / "jammi-cookbook"))

# Pinned sources: (url, sha256). Air Routes is pinned to one immutable commit
# so its node and edge files are mutually consistent (graph version 0.89).
_AIR_COMMIT = "efd3b1ae636f602577cfbccb16ecfe358a02ee36"
_AIR_BASE = f"https://raw.githubusercontent.com/krlawrence/graph/{_AIR_COMMIT}/sample-data"
_AIR_NODES = (
    f"{_AIR_BASE}/air-routes-latest-nodes.csv",
    "f921d4f1dd429418a96c17c49d42b42ce6cf5d6c9772e784556a631760b53579",
)
_AIR_EDGES = (
    f"{_AIR_BASE}/air-routes-latest-edges.csv",
    "01749b2717ccca5efe11c4b1f5e25f8c59ab682014104f3fb8bdb67e23b101b5",
)
_ARXIV_ZIP = (
    "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
    "49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276",
)
_ARXIV_TITLEABS = (
    "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz",
    "7bce99ab3e1604277f12dd49f6e17a0d89867b29ea152f072c0e709ae0bc8ed7",
)

LICENSES = {
    "air_routes": "Permissive (krlawrence/graph sample-data); see NOTICE.",
    "ogbn_arxiv": "ODC-BY 1.0 (Open Graph Benchmark); see NOTICE.",
}


# --------------------------------------------------------------------------- #
# Registered datasets
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AirRoutes:
    """The registered Air Routes sources."""

    airports: str
    routes: str
    contains: str


@dataclass(frozen=True)
class Arxiv:
    """The registered ogbn-arxiv sources and the ball's time split."""

    papers: str
    cites: str
    split: dict[str, list[str]]  # "train" / "valid" / "test" → paper_ids


def air_routes(db) -> AirRoutes:
    """Register Air Routes into ``db``: ``air_airports`` (key ``code``, text
    ``desc``, the ``continent`` label), ``air_routes`` (``src``, ``dst``,
    ``dist``) and ``air_contains`` (``src`` parent, ``dst`` child). Each file
    is named for its source, so each queries as ``<source>.public.<source>``."""
    registered = AirRoutes("air_airports", "air_routes", "air_contains")
    for name in (registered.airports, registered.routes, registered.contains):
        db.add_source(name, url=fixtures.url(f"air_routes/{name}.parquet"), format="parquet")
    return registered


def arxiv(db, scale: Scale) -> Arxiv:
    """Register the scale's ogbn-arxiv ball into ``db``: ``arxiv_papers`` (key
    ``paper_id``, text ``title`` + ``abstract``, label ``subject``, ``year``)
    and ``arxiv_cites`` (``src`` cites ``dst``)."""
    if scale is Scale.SMALL:
        papers_url = fixtures.url("arxiv_small/arxiv_papers.parquet")
        cites_url = fixtures.url("arxiv_small/arxiv_cites.parquet")
        papers = pq.read_table(fixtures.path("arxiv_small/arxiv_papers.parquet"))
    else:
        papers, cites = arxiv_tables(scale)
        papers_url = _cached_parquet(papers, f"{scale}/arxiv_papers")
        cites_url = _cached_parquet(cites, f"{scale}/arxiv_cites")
    db.add_source("arxiv_papers", url=papers_url, format="parquet")
    db.add_source("arxiv_cites", url=cites_url, format="parquet")
    return Arxiv("arxiv_papers", "arxiv_cites", time_split(papers))


def time_split(papers: pa.Table) -> dict[str, list[str]]:
    """The ogbn-arxiv time split of ``papers``, by year."""
    split: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    ids, years = papers.column("paper_id").to_pylist(), papers.column("year").to_pylist()
    for pid, year in zip(ids, years, strict=True):
        name = "train" if year <= TRAIN_UNTIL else "valid" if year == VALID_YEAR else "test"
        split[name].append(pid)
    return split


def same_label_golden(
    db, rows: list[dict], *, key: str, label: str, text: str, queries: int, name: str
) -> str:
    """Register a retrieval golden where a row's relevant rows are the other
    rows sharing its ``label`` — a target independent of any embedding, so an
    embedding that retrieves it better is better, not circular. The first
    ``queries`` rows (in key order) whose label has at least five members are
    the queries, each asked by its ``text``. Returns the golden's source name.
    """
    members: dict[str, list[str]] = {}
    for r in rows:
        members.setdefault(r[label], []).append(r[key])
    asked = [r for r in sorted(rows, key=lambda r: r[key]) if len(members[r[label]]) >= 5]
    golden = [
        {"query_id": q[key], "query_text": q[text], "relevant_id": other}
        for q in asked[:queries]
        for other in members[q[label]]
        if other != q[key]
    ]
    url = _cached_parquet(pa.Table.from_pylist(golden), f"goldens/{name}")
    db.add_source(name, url=url, format="parquet")
    return f"{name}.public.{name}"


# --------------------------------------------------------------------------- #
# Building the tables from the pinned sources
# --------------------------------------------------------------------------- #


def arxiv_tables(scale: Scale) -> tuple[pa.Table, pa.Table]:
    """The scale's papers and the citations among them, from the pinned
    archive: papers ``(paper_id, title, abstract, subject, year)`` in ball
    order. ``full`` is the :data:`ARXIV_BALL`-paper breadth-first ball;
    ``small`` is its :data:`ARXIV_CORE` papers with the most citations inside
    the ball (ties to ball order)."""
    zipped = _download(*_ARXIV_ZIP, name="arxiv.zip")
    with zipfile.ZipFile(zipped) as zf:

        def lines(member: str) -> list[str]:
            with zf.open(member) as f:
                return gzip.decompress(f.read()).decode("utf-8").splitlines()

        num_nodes = int(lines("arxiv/raw/num-node-list.csv.gz")[0])
        edges = [
            (int(a), int(b)) for a, b in (ln.split(",") for ln in lines("arxiv/raw/edge.csv.gz"))
        ]
        labels = [int(x) for x in lines("arxiv/raw/node-label.csv.gz")]
        years = [int(x) for x in lines("arxiv/raw/node_year.csv.gz")]
        node2pid = [
            int(ln.split(",")[1]) for ln in lines("arxiv/mapping/nodeidx2paperid.csv.gz")[1:]
        ]
        subjects = [
            ln.split(",", 1)[1] for ln in lines("arxiv/mapping/labelidx2arxivcategeory.csv.gz")[1:]
        ]
    text = _titleabs()

    ball = _ball(num_nodes, edges, ARXIV_BALL)
    if scale is Scale.SMALL:
        ball = _core(ball, edges, ARXIV_CORE)
    members = set(ball)
    papers = pa.table(
        {
            "paper_id": [str(node2pid[n]) for n in ball],
            "title": [text[node2pid[n]][0] for n in ball],
            "abstract": [text[node2pid[n]][1] for n in ball],
            "subject": [subjects[labels[n]] for n in ball],
            "year": pa.array([years[n] for n in ball], pa.int64()),
        }
    )
    cited = [(s, d) for s, d in edges if s in members and d in members]
    cites = pa.table(
        {
            "src": [str(node2pid[s]) for s, _ in cited],
            "dst": [str(node2pid[d]) for _, d in cited],
        }
    )
    return papers, cites


def _ball(num_nodes: int, edges: list[tuple[int, int]], size: int) -> list[int]:
    """``size`` node indices collected breadth-first over the undirected
    citation graph from the highest-degree node (ties to the lowest index),
    neighbours in index order. A pure function of the graph: every prefix of a
    larger ball is the smaller ball."""
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for s, d in edges:
        adj[s].append(d)
        adj[d].append(s)
    start = max(range(num_nodes), key=lambda i: (len(adj[i]), -i))
    seen, order, queue = {start}, [start], deque([start])
    while queue and len(order) < size:
        for nbr in sorted(adj[queue.popleft()]):
            if nbr not in seen:
                seen.add(nbr)
                order.append(nbr)
                queue.append(nbr)
                if len(order) == size:
                    break
    return order


def _core(ball: list[int], edges: list[tuple[int, int]], size: int) -> list[int]:
    """The ``size`` nodes of ``ball`` with the most edges inside it, in ball
    order."""
    inside = set(ball)
    degree = {n: 0 for n in ball}
    for s, d in edges:
        if s in inside and d in inside:
            degree[s] += 1
            degree[d] += 1
    position = {n: i for i, n in enumerate(ball)}
    kept = set(sorted(ball, key=lambda n: (-degree[n], position[n]))[:size])
    return [n for n in ball if n in kept]


def _titleabs() -> dict[int, tuple[str, str]]:
    """paper_id → (title, abstract)."""
    path = _download(*_ARXIV_TITLEABS, name="titleabs.tsv.gz")
    text: dict[int, tuple[str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and parts[0].strip().isdigit():
                text[int(parts[0])] = (parts[1], parts[2])
    return text


def air_routes_tables() -> tuple[pa.Table, pa.Table, pa.Table]:
    """Airports, routes and the contains hierarchy, from the pinned CSVs. An
    airport's ``continent`` is its parent in the continent→airport edges."""
    nodes_csv = _download(*_AIR_NODES, name="air-routes-nodes.csv")
    edges_csv = _download(*_AIR_EDGES, name="air-routes-edges.csv")
    label_code: dict[str, tuple[str, str]] = {}
    airports: dict[str, dict] = {}
    with nodes_csv.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            nid, label, _type, code, _icao, desc, region = row[0:7]
            runways, longest, elev, country, city, lat, lon = row[7:14]
            label_code[nid] = (label, code)
            if label == "airport":
                airports[code] = {
                    "code": code, "desc": desc, "city": city, "country": country,
                    "continent": "", "lat": float(lat), "lon": float(lon),
                    "elev": int(elev), "runways": int(runways), "longest": int(longest),
                    "region": region,
                }
    routes: list[tuple[str, str, int]] = []
    contains: list[tuple[str, str]] = []
    with edges_csv.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src_label, src = label_code[row[1]]
            _, dst = label_code[row[2]]
            if row[3] == "route":
                routes.append((src, dst, int(row[4])))
            elif row[3] == "contains":
                contains.append((src, dst))
                if src_label == "continent" and dst in airports:
                    airports[dst]["continent"] = src
    return (
        pa.Table.from_pylist([airports[c] for c in sorted(airports)]),
        pa.table(
            {
                "src": [s for s, _, _ in routes],
                "dst": [d for _, d, _ in routes],
                "dist": pa.array([w for _, _, w in routes], pa.int64()),
            }
        ),
        pa.table({"src": [s for s, _ in contains], "dst": [d for _, d in contains]}),
    )


def write_fixtures() -> None:
    """Author-time: rewrite the committed ``air_routes`` and ``arxiv_small``
    fixtures from the pinned sources."""
    airports, routes, contains = air_routes_tables()
    air = fixtures.path("air_routes")
    for table, name in ((airports, "air_airports"), (routes, "air_routes"),
                        (contains, "air_contains")):
        pq.write_table(table, air / f"{name}.parquet")
    papers, cites = arxiv_tables(Scale.SMALL)
    small = fixtures.path("arxiv_small")
    pq.write_table(papers, small / "arxiv_papers.parquet")
    pq.write_table(cites, small / "arxiv_cites.parquet")


# --------------------------------------------------------------------------- #
# Downloads
# --------------------------------------------------------------------------- #

# A dropped connection, a stalled read, or a 429/5xx from a source's host is
# retried with backoff rather than failing the fetch.
_RETRY = Retry(
    total=6,
    backoff_factor=2.0,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
)


def _download(url: str, sha256: str, *, name: str) -> Path:
    """``url``, fetched once into the cache and verified against ``sha256``."""
    dest = _CACHE / "raw" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=_RETRY))
            session.mount("http://", HTTPAdapter(max_retries=_RETRY))
            resp = session.get(url, timeout=(30, 300))
            resp.raise_for_status()
            dest.write_bytes(resp.content)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != sha256:
        dest.unlink()
        raise ValueError(
            f"checksum mismatch for {url}: expected {sha256}, got {digest}. The pinned "
            "source changed; the book refuses it rather than drift its numbers."
        )
    return dest


def _cached_parquet(table: pa.Table, name: str) -> str:
    """``table`` written to the cache, as the URL ``add_source`` registers."""
    path = _CACHE / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path.as_uri()


if __name__ == "__main__":
    write_fixtures()
