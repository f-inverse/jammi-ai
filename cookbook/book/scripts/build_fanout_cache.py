"""Emit the fan-out chapter's cache: one embedding plan, run at several
partition counts, writes the same bytes.

The rows a model forwards together are decided once, by row cost, and carried
as a chunk id the plan's exchange hashes on; so a plan fanned over one
partition or four forwards identical chunks and must write an identical
artifact. This script runs the same `generate_embeddings` over the same corpus
at each fan-out in `PARTITIONS` — each in its own fresh `file://` catalog,
configured through `[inference] partitions` — and records, per run, the
artifact digest and the definition hash off the table's manifest sidecar.

The corpus is a committed literal whose rows differ widely in length, so the
row-cost order and the chunk cut are exercised; `batch_size = 4` makes every
fan-out forward several chunks.

Usage::

    python scripts/build_fanout_cache.py --fixtures-root /path/to/engine/checkout
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import jammi
import pyarrow as pa
import pyarrow.parquet as pq

import jammi_cookbook  # noqa: F401  # applies the determinism env on import
from jammi_cookbook import cache

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "fanout"

PARTITIONS = (1, 2, 4)
BATCH_SIZE = 4

# 24 rows, one to fourteen words each: the lengths are what the chunk cut
# orders on, so they are deliberately uneven.
_DOCS = {
    "_row_id": [f"d{i:02d}" for i in range(24)],
    "text": [
        "graph",
        "signal processing on graphs",
        "a spectral filter applied to node features over the normalized laplacian",
        "message passing",
        "random walks with restart as a personalized ranking",
        "edge weights",
        "attention over neighbours weights each message by a learned compatibility score",
        "pooling",
        "a laplacian eigenmap embeds nodes by the smallest nontrivial eigenvectors",
        "k nearest neighbours",
        "propagation smooths a signal along the edges of the graph",
        "label spreading",
        "a node embedding",
        "the adjacency matrix and its powers count walks between node pairs",
        "diffusion",
        "graph convolution as a first order polynomial of the laplacian",
        "readout",
        "a heterogeneous graph carries typed nodes and typed edges",
        "clustering coefficient",
        "spectral clustering partitions the graph by the fiedler vector of its laplacian",
        "homophily",
        "over smoothing makes deep propagation collapse node representations together",
        "subgraph sampling",
        "the incidence matrix relates edges to their endpoints",
    ],
}


def _sidecar(catalog_root: Path, table: str) -> dict:
    """The table's `.materialization.json`, read beside its Parquet object."""
    path = catalog_root / "jammi_db" / "_global" / f"{table}.materialization.json"
    return json.loads(path.read_text())


def embed_at(work: Path, partitions: int, model: str, source: Path) -> dict:
    """Embed `source` in a fresh catalog fanned over `partitions`."""
    root = work / f"p{partitions}"
    root.mkdir()
    config = root / "jammi.toml"
    config.write_text(
        f"[gpu]\ndevice = -1\n\n[inference]\npartitions = {partitions}\n"
        f"batch_size = {BATCH_SIZE}\n"
    )
    catalog = root / "catalog"
    with jammi.connect(f"file://{catalog}", config=str(config)) as db:
        db.add_source("docs", url=str(source), format="parquet")
        table = db.generate_embeddings(
            source="docs", model=model, columns=["text"], key="_row_id", modality="text"
        )
    manifest = _sidecar(catalog, table)
    return {
        "partitions": partitions,
        "artifact": manifest["artifact"],
        "definition_hash": manifest["definition_hash"],
    }


def emit(fixtures_root: Path) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model = f"local:{fixtures_root / 'tests' / 'fixtures' / 'tiny_modernbert'}"
    with tempfile.TemporaryDirectory() as work_root:
        work = Path(work_root)
        source = work / "docs.parquet"
        pq.write_table(pa.table(_DOCS), source)
        runs = [embed_at(work, n, model, source) for n in PARTITIONS]

    reference = runs[0]
    matrix = {
        "runs": runs,
        "rows": len(_DOCS["_row_id"]),
        "batch_size": BATCH_SIZE,
    }
    golden = {
        f"p{run['partitions']}_artifact_equal": {
            "value": 1.0 if run["artifact"] == reference["artifact"] else 0.0,
            "tol": 0.0,
        }
        for run in runs[1:]
    } | {
        "definition_equal_all": {
            "value": 1.0
            if all(run["definition_hash"] == reference["definition_hash"] for run in runs)
            else 0.0,
            "tol": 0.0,
        },
    }
    record = {
        "method": (
            "generate_embeddings over one committed 24-row corpus, each fan-out in its own "
            "fresh file:// catalog configured with [inference] partitions = N and "
            f"batch_size = {BATCH_SIZE}; the artifact digest and definition hash are read "
            "off each table's .materialization.json sidecar."
        ),
        "partitions": list(PARTITIONS),
        "engine_version": jammi.__version__,
        "model": "tests/fixtures/tiny_modernbert",
    }

    (ARTIFACTS / "matrix.json").write_text(json.dumps(matrix, indent=2, sort_keys=True))
    (ARTIFACTS / "golden_metrics.json").write_text(json.dumps(golden, indent=2, sort_keys=True))
    (ARTIFACTS / "fanout.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    cache.write_checksums(ARTIFACTS)

    print("\n=== one plan, every fan-out, the same bytes ===", flush=True)
    for run in runs:
        print(f"  partitions={run['partitions']}  artifact={run['artifact']}", flush=True)
    for metric, verdict in sorted(golden.items()):
        print(f"  {metric} = {verdict['value']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fixtures-root",
        default=None,
        help="engine checkout with tests/fixtures/tiny_modernbert (or set JAMMI_FIXTURES_ROOT)",
    )
    args = ap.parse_args()
    fixtures = cache.engine_fixtures_root(
        args.fixtures_root, "tests/fixtures/tiny_modernbert/config.json"
    )
    emit(fixtures)


if __name__ == "__main__":
    main()
