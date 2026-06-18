"""Dataset loaders — Air Routes and ogbn-arxiv (K2).

Both datasets are publicly redistributable and are fetched on demand, never
committed in full: only the small committed subset artifacts (``artifacts/``) and
the subset ID lists (``data/ids/``) live in the repo. Every download is
checksum-gated against a pinned digest — a tampered or silently-reissued source
fails loudly rather than drifting the book's numbers.

* **Air Routes** (Neptune's own teaching dataset; permissive, from
  ``krlawrence/graph``) — airports + ``route`` / ``contains`` edges. The tiers
  01–02 on-ramp.
* **ogbn-arxiv** (ODC-BY; Open Graph Benchmark) — ~169k CS papers, ~1.16M
  citation edges, 40 subject classes, title+abstract text. The tiers 03–04 spine.

The loaders register file-shaped sources into a ``jammi_ai`` database and return
the committed subset; subset identity comes from the committed ID lists
(:func:`jammi_cookbook.determinism.committed_ids`), not from replaying a seed.
Both datasets are read with the standard library + pyarrow from their pinned,
checksum-gated archives — no torch or graph-library dependency.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from . import determinism

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RAW_DIR = _REPO_ROOT / "data" / "raw"
_CACHE_DIR = _REPO_ROOT / "data" / "cache"
_IDS_DIR = _REPO_ROOT / "data" / "ids"


# --------------------------------------------------------------------------- #
# Pinned sources (the determinism contract for downloads)
# --------------------------------------------------------------------------- #

# Air Routes is pinned to a single immutable repo commit so the node and edge
# files are mutually consistent (graph version 0.89); the moving `master` raw URL
# served stale CDN content, which is exactly what the checksum gate catches.
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

# ogbn-arxiv: the canonical graph/labels/year/split + id/label mappings ship as
# gzip'd CSVs inside the Open Graph Benchmark zip, read directly (no torch/ogb
# dependency). The raw title+abstract text is a separate file. Both are
# checksum-gated, so determinism is a property of the pinned digest.
_ARXIV_ZIP = (
    "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip",
    "49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276",
)
_ARXIV_TITLEABS = (
    "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz",
    "7bce99ab3e1604277f12dd49f6e17a0d89867b29ea152f072c0e709ae0bc8ed7",
)

# Licenses, recorded here and in NOTICE.
LICENSES = {
    "air_routes": "Permissive (krlawrence/graph sample-data); see NOTICE.",
    "ogbn_arxiv": "ODC-BY 1.0 (Open Graph Benchmark); see NOTICE.",
}


def _download(url: str, sha256: str, *, dest: Path) -> Path:
    """Fetch ``url`` to ``dest`` (cached) and verify its SHA-256 digest.

    A digest mismatch raises — a changed or tampered source must fail, never
    silently reshape the data the book is pinned to.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != sha256:
        dest.unlink(missing_ok=True)
        raise ValueError(
            f"checksum mismatch for {url}\n  expected {sha256}\n  got      {digest}\n"
            f"The pinned source changed; the book's determinism contract refuses it."
        )
    return dest


def _write_parquet(table: pa.Table, name: str) -> str:
    """Write a table to the (gitignored) cache and return its path as a URL."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{name}.parquet"
    pq.write_table(table, path)
    return str(path)


# --------------------------------------------------------------------------- #
# Air Routes
# --------------------------------------------------------------------------- #

# The airport node columns the book uses (key first). `continent` is derived from
# the `contains` hierarchy (continent→airport edges), not a raw column.
_AIRPORT_COLUMNS = [
    "code",
    "desc",
    "city",
    "country",
    "continent",
    "lat",
    "lon",
    "elev",
    "runways",
    "longest",
    "region",
]


@dataclass(frozen=True)
class AirRoutes:
    """Registered Air Routes sources + the loaded airport subset."""

    airports_source: str
    route_edges_source: str
    contains_edges_source: str
    airport_codes: list[str]


def _parse_air_routes() -> tuple[
    dict[str, dict], list[tuple[str, str, int]], list[tuple[str, str]]
]:
    """Parse the pinned CSVs into (airports-by-code, route edges, contains edges).

    Returns airports keyed by IATA ``code`` with each airport's ``continent``
    resolved from the continent→airport ``contains`` edges; ``route`` edges as
    ``(src_code, dst_code, dist)``; and the full ``contains`` hierarchy as
    ``(parent_code, child_code)``.
    """
    nodes_csv = _download(*_AIR_NODES, dest=_RAW_DIR / "air-routes-nodes.csv")
    edges_csv = _download(*_AIR_EDGES, dest=_RAW_DIR / "air-routes-edges.csv")

    # id -> (label, code) and the airport rows by code.
    id_label_code: dict[str, tuple[str, str]] = {}
    airports: dict[str, dict] = {}
    with nodes_csv.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            nid, label, _type, code, _icao, desc, region = row[0:7]
            runways, longest, elev, country, city, lat, lon = row[7:14]
            id_label_code[nid] = (label, code)
            if label == "airport":
                airports[code] = {
                    "code": code,
                    "desc": desc,
                    "city": city,
                    "country": country,
                    "continent": "",
                    "lat": float(lat),
                    "lon": float(lon),
                    "elev": int(elev),
                    "runways": int(runways),
                    "longest": int(longest),
                    "region": region,
                }

    route_edges: list[tuple[str, str, int]] = []
    contains_edges: list[tuple[str, str]] = []
    with edges_csv.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            _id, frm, to, label = row[0:4]
            src_label, src_code = id_label_code[frm]
            _dst_label, dst_code = id_label_code[to]
            if label == "route":
                route_edges.append((src_code, dst_code, int(row[4])))
            elif label == "contains":
                contains_edges.append((src_code, dst_code))
                # continent→airport edge resolves the airport's continent.
                if src_label == "continent" and dst_code in airports:
                    airports[dst_code]["continent"] = src_code

    return airports, route_edges, contains_edges


def load_air_routes(db) -> AirRoutes:
    """Register the Air Routes sources into ``db`` and return the airport subset.

    The full small graph is loaded: every committed airport plus its incident
    ``route`` and ``contains`` edges. Subset identity is the committed
    ``data/ids/air.txt`` list (the full airport set). License: see
    :data:`LICENSES`.
    """
    airports, route_edges, contains_edges = _parse_air_routes()

    committed = set(determinism.committed_ids("air"))
    keep = {c for c in airports if c in committed}

    airport_rows = [airports[c] for c in sorted(keep)]
    airports_table = pa.Table.from_pylist(airport_rows, schema=_airport_schema())
    routes_table = pa.table(
        {
            "src": [s for s, d, _ in route_edges if s in keep and d in keep],
            "dst": [d for s, d, _ in route_edges if s in keep and d in keep],
            "dist": [w for s, d, w in route_edges if s in keep and d in keep],
        }
    )
    contains_table = pa.table(
        {
            "src": [s for s, d in contains_edges if d in keep],
            "dst": [d for s, d in contains_edges if d in keep],
        }
    )

    airports_source = "air_airports"
    route_source = "air_route_edges"
    contains_source = "air_contains_edges"
    db.add_source(
        airports_source, url=_write_parquet(airports_table, airports_source), format="parquet"
    )
    db.add_source(route_source, url=_write_parquet(routes_table, route_source), format="parquet")
    db.add_source(
        contains_source, url=_write_parquet(contains_table, contains_source), format="parquet"
    )

    return AirRoutes(
        airports_source=airports_source,
        route_edges_source=route_source,
        contains_edges_source=contains_source,
        airport_codes=sorted(keep),
    )


def _airport_schema() -> pa.Schema:
    return pa.schema(
        [
            ("code", pa.string()),
            ("desc", pa.string()),
            ("city", pa.string()),
            ("country", pa.string()),
            ("continent", pa.string()),
            ("lat", pa.float64()),
            ("lon", pa.float64()),
            ("elev", pa.int64()),
            ("runways", pa.int64()),
            ("longest", pa.int64()),
            ("region", pa.string()),
        ]
    )


def write_air_routes_ids() -> list[str]:
    """Author-time helper: write ``data/ids/air.txt`` with every airport code.

    Air Routes is small enough to run as the full graph, so the committed subset
    is all airports; recorded for the determinism contract (committed, not seeded).
    """
    airports, _, _ = _parse_air_routes()
    codes = sorted(airports)
    _IDS_DIR.mkdir(parents=True, exist_ok=True)
    (_IDS_DIR / "air.txt").write_text("\n".join(codes) + "\n")
    return codes


# --------------------------------------------------------------------------- #
# ogbn-arxiv
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OgbnArxiv:
    """Registered ogbn-arxiv sources + the committed subset and its time-split."""

    papers_source: str
    cite_edges_source: str
    paper_ids: list[str]
    split: dict[str, list[str]]  # "train" / "valid" / "test" → paper_ids in subset


@dataclass(frozen=True)
class _ArxivRaw:
    """The canonical ogbn-arxiv tables, read straight from the pinned zip."""

    num_nodes: int
    edges: list[tuple[int, int]]  # directed citation edges, node indices
    labels: list[int]  # subject-class index per node
    years: list[int]  # publication year per node
    node2pid: list[int]  # node index → MAG paper id
    label_names: list[str]  # class index → arxiv category name
    split: dict[str, list[int]]  # canonical time-split, node indices


def _zip_member_lines(zf, member: str) -> list[str]:
    import gzip as _gz

    with zf.open(member) as f:
        return _gz.decompress(f.read()).decode("utf-8").splitlines()


def _load_arxiv_raw() -> _ArxivRaw:
    """Read the canonical ogbn-arxiv tables directly from the checksum-gated zip.

    The Open Graph Benchmark distributes the full graph, labels, years, time-split,
    and id/label mappings as gzip'd CSVs inside ``arxiv.zip``. Reading them
    directly — rather than through the ``ogb`` package — keeps the loader free of a
    heavy, torch-pinned dependency and makes determinism a property of the pinned
    checksum, not of a pickle cache.
    """
    import zipfile

    path = _download(*_ARXIV_ZIP, dest=_RAW_DIR / "arxiv.zip")
    with zipfile.ZipFile(path) as zf:
        num_nodes = int(_zip_member_lines(zf, "arxiv/raw/num-node-list.csv.gz")[0])
        edges = [
            (int(a), int(b))
            for a, b in (line.split(",") for line in _zip_member_lines(zf, "arxiv/raw/edge.csv.gz"))
        ]
        labels = [int(x) for x in _zip_member_lines(zf, "arxiv/raw/node-label.csv.gz")]
        years = [int(x) for x in _zip_member_lines(zf, "arxiv/raw/node_year.csv.gz")]
        node2pid = [
            int(line.split(",")[1])
            for line in _zip_member_lines(zf, "arxiv/mapping/nodeidx2paperid.csv.gz")[1:]
        ]
        label_names = [
            line.split(",", 1)[1]
            for line in _zip_member_lines(zf, "arxiv/mapping/labelidx2arxivcategeory.csv.gz")[1:]
        ]
        split = {
            name: [int(x) for x in _zip_member_lines(zf, f"arxiv/split/time/{name}.csv.gz")]
            for name in ("train", "valid", "test")
        }
    return _ArxivRaw(num_nodes, edges, labels, years, node2pid, label_names, split)


def _load_titleabs() -> dict[int, tuple[str, str]]:
    """paper_id → (title, abstract), from the checksum-gated titleabs file."""
    path = _download(*_ARXIV_TITLEABS, dest=_RAW_DIR / "titleabs.tsv.gz")
    text: dict[int, tuple[str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3 or not parts[0].strip().isdigit():
                continue  # drops the stray first-line filename artifact
            text[int(parts[0])] = (parts[1], parts[2])
    return text


def _connected_subset(raw: _ArxivRaw, size: int) -> list[int]:
    """A deterministic connected induced subgraph of ``size`` node indices.

    BFS over the undirected citation graph from the highest-degree node (ties
    broken by lowest index) until ``size`` nodes are collected. A connected ball
    preserves citation density and label homophily — the structure the tier-04
    coverage crux depends on. A pure function of the graph: no seed needed.
    """
    adj: list[list[int]] = [[] for _ in range(raw.num_nodes)]
    for s, d in raw.edges:
        adj[s].append(d)
        adj[d].append(s)
    start = max(range(raw.num_nodes), key=lambda i: (len(adj[i]), -i))

    seen = {start}
    order = [start]
    queue = deque([start])
    while queue and len(order) < size:
        node = queue.popleft()
        for nbr in sorted(adj[node]):  # sorted → deterministic expansion
            if nbr not in seen:
                seen.add(nbr)
                order.append(nbr)
                queue.append(nbr)
                if len(order) >= size:
                    break
    return order[:size]


def load_ogbn_arxiv(db, *, subset: int = 4000) -> OgbnArxiv:
    """Register the ogbn-arxiv sources into ``db`` and return the committed subset.

    The papers source has ``paper_id`` (key), ``title``, ``abstract`` (the text to
    embed), ``subject`` (the 40-class label name), and ``year`` (the date split).
    The ``cite_edges`` source is the declared citation graph — the BYOG signal for
    tier-04. The subset is the committed ``data/ids/arxiv.txt`` list (a connected
    induced subgraph); ``subset`` sizes it only on first author-time generation.
    License: see :data:`LICENSES`.
    """
    raw = _load_arxiv_raw()
    text = _load_titleabs()

    pid_to_node = {raw.node2pid[i]: i for i in range(raw.num_nodes)}
    committed = _committed_arxiv_ids(raw, subset)
    keep_pids = [int(p) for p in committed]
    keep_nodes = {pid_to_node[p] for p in keep_pids}

    papers_rows = []
    for pid in keep_pids:
        node = pid_to_node[pid]
        title, abstract = text[pid]
        papers_rows.append(
            {
                "paper_id": str(pid),
                "title": title,
                "abstract": abstract,
                "subject": raw.label_names[raw.labels[node]],
                "year": raw.years[node],
            }
        )
    papers_table = pa.Table.from_pylist(papers_rows, schema=_paper_schema())

    cite_table = pa.table(
        {
            "src": [
                str(raw.node2pid[s]) for s, d in raw.edges if s in keep_nodes and d in keep_nodes
            ],
            "dst": [
                str(raw.node2pid[d]) for s, d in raw.edges if s in keep_nodes and d in keep_nodes
            ],
        }
    )

    db.add_source(
        "arxiv_papers", url=_write_parquet(papers_table, "arxiv_papers"), format="parquet"
    )
    db.add_source(
        "arxiv_cite_edges", url=_write_parquet(cite_table, "arxiv_cite_edges"), format="parquet"
    )

    keep_set = set(keep_pids)
    split_pids = {
        name: sorted(str(raw.node2pid[i]) for i in idx if raw.node2pid[i] in keep_set)
        for name, idx in raw.split.items()
    }

    return OgbnArxiv(
        papers_source="arxiv_papers",
        cite_edges_source="arxiv_cite_edges",
        paper_ids=[str(p) for p in keep_pids],
        split=split_pids,
    )


def _committed_arxiv_ids(raw: _ArxivRaw, subset: int) -> list[str]:
    """The committed arxiv subset paper_ids, generating + writing them once.

    On first author-time call (no committed list) the connected subgraph is
    selected and written to ``data/ids/arxiv.txt`` as the source of truth; every
    later call (including CI) reads that file.
    """
    path = _IDS_DIR / "arxiv.txt"
    if path.exists():
        return determinism.committed_ids("arxiv")
    pids = [str(raw.node2pid[i]) for i in _connected_subset(raw, subset)]
    _IDS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pids) + "\n")
    return pids


def _paper_schema() -> pa.Schema:
    return pa.schema(
        [
            ("paper_id", pa.string()),
            ("title", pa.string()),
            ("abstract", pa.string()),
            ("subject", pa.string()),
            ("year", pa.int64()),
        ]
    )
