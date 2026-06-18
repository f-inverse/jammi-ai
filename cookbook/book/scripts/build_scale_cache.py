#!/usr/bin/env python3
"""Emit the ogbn-arxiv SCALE-tier vector cache (H2 scale tier) — run ONCE on the GPU server.

This is the heavy embedding pass at scale: the FULL ogbn-arxiv graph (every node
that carries a title+abstract) is embedded with ModernBERT on the GPU, the
``(_row_id, vector)`` matrix is pulled back, and a HELD-OUT recall split is
written. It is the scale sibling of ``build_arxiv_cache.py`` (the 4 000-paper
keystone) and does NOT mutate that keystone or its committed ``data/ids/arxiv.txt``.

What this script produces (under ``artifacts/scale/`` + ``data/ids/``):

* ``data/ids/arxiv_scale.txt``      — the committed id list: every embedded paper_id.
* ``scale_queries.txt``             — the HELD-OUT query paper_ids (the recall probes).
* ``arxiv_vectors.parquet``         — the CORPUS (indexed) vectors: ``(_row_id, vector)``,
                                      768-dim f32, the N−Q papers NOT held out.
* ``scale_query_vectors.parquet``   — the HELD-OUT query vectors: ``(_row_id, vector)``.

The ANN frozen index, the exact ground truth, the recall curve over the held-out
queries, and ``golden_metrics.json`` are emitted by the engine's ``jammi-bench``
recall path (RC1: the sidecar is not pullable over gRPC, so it is rebuilt-and-
frozen ONCE on this box by the engine's own SidecarIndex builder, never in CI).

The held-out split is mandatory (design v2.1): the recall floor MUST be measured
over queries that are NOT in the indexed corpus, so each query's true nearest
neighbour is a DIFFERENT paper and recall@1 is meaningful — a corpus-as-its-own-
query split would yield a structural recall@1 ≈ 1.0 that floors nothing.

The split is deterministic and committed by id, not by seed: sort every embedded
paper_id ascending and hold out the LAST ``QUERY_COUNT`` of them as queries; the
rest form the corpus. The same partition reproduces on any box from the committed
id lists alone.

Usage::

    # 1. start the GPU server (bare process, NOT docker --gpus; §7.3):
    #    JAMMI_ARTIFACT_DIR=/scratch JAMMI_GPU__DEVICE=0 JAMMI_GPU__REQUIRE_GPU=true \\
    #    JAMMI_SERVER__FLIGHT_LISTEN=127.0.0.1:50051 RAYON_NUM_THREADS=1 jammi-server &
    # 2. emit against it:
    python scripts/build_scale_cache.py --target grpc://127.0.0.1:50051
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jammi_ai
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import jammi_cookbook  # noqa: F401  # applies the determinism env on import
from jammi_cookbook import datasets

EMBED_MODEL = "answerdotai/ModernBERT-base"
REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts" / "scale"
IDS_DIR = REPO_ROOT / "data" / "ids"

# Hold out this many papers (by sorted paper_id) as recall queries. They are NOT
# indexed; their true nearest neighbour is a different paper, so recall@1 is a
# real floor. 1 000 over a ~170k corpus is a stable mean while leaving the corpus
# essentially full.
QUERY_COUNT = 1000


def _full_papers_table() -> tuple[pa.Table, list[str]]:
    """The FULL ogbn-arxiv papers table: every node that has a title+abstract.

    Reads the canonical graph + the title/abstract text straight from the
    checksum-gated raw files (the same loaders ``datasets.load_ogbn_arxiv`` uses),
    keeping every node whose paper_id appears in the title/abstract file. This is
    the full embeddable corpus — not the keystone's 4 000-node connected ball — so
    it must build its own papers table rather than route through
    ``load_ogbn_arxiv`` (which is pinned to the committed 4k id list).

    Returns the papers table (paper_id/title/abstract/subject/year) and the
    paper_ids in node order.
    """
    raw = datasets._load_arxiv_raw()
    text = datasets._load_titleabs()

    rows = []
    for node in range(raw.num_nodes):
        pid = raw.node2pid[node]
        ta = text.get(pid)
        if ta is None:
            continue  # no embeddable text → not part of the corpus
        title, abstract = ta
        rows.append(
            {
                "paper_id": str(pid),
                "title": title,
                "abstract": abstract,
                "subject": raw.label_names[raw.labels[node]],
                "year": raw.years[node],
            }
        )
    table = pa.Table.from_pylist(rows, schema=datasets._paper_schema())
    return table, [r["paper_id"] for r in rows]


def _read_vectors(db, table: str) -> tuple[list[str], np.ndarray]:
    """Read an engine embedding table's ``(_row_id, vector)`` sorted by _row_id."""
    ref = f'"jammi.{table}"'
    t = db.sql(f"SELECT _row_id, vector FROM {ref} ORDER BY _row_id")
    ids = [str(x) for x in t.column("_row_id").to_pylist()]
    vecs = np.asarray([list(v) for v in t.column("vector").to_pylist()], dtype=np.float32)
    return ids, vecs


def _write_vectors(path: Path, ids: list[str], vecs: np.ndarray) -> None:
    """Write ``(_row_id, vector)`` as the engine embedding-table parquet shape.

    ``_row_id`` is a string key; ``vector`` is a fixed-size list of f32 of the
    embedding width — the shape the engine's corpus reader
    (``extend_with_fixed_size_list_f32``) and the exact oracle scan expect.
    """
    dim = int(vecs.shape[1])
    vec_type = pa.list_(pa.field("item", pa.float32(), nullable=False), dim)
    table = pa.table(
        {
            "_row_id": pa.array(ids, type=pa.string()),
            "vector": pa.array((v for v in vecs), type=vec_type),
        }
    )
    pq.write_table(table, path, compression="zstd")


def emit(db) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    IDS_DIR.mkdir(parents=True, exist_ok=True)
    info = db.get_server_info()
    print("server:", json.dumps(info), flush=True)

    papers_table, paper_ids = _full_papers_table()
    print(f"full corpus: {len(paper_ids)} embeddable papers", flush=True)

    # Register the full papers source (its own parquet in the gitignored cache —
    # the keystone's arxiv_papers source is the 4k subset, so use a distinct name).
    url = datasets._write_parquet(papers_table, "arxiv_scale_papers")
    db.add_source("arxiv_scale_papers", url=url, format="parquet")

    # ---- the heavy GPU pass: embed title+abstract with ModernBERT ----
    print(f"\n[embed] generate_embeddings on {EMBED_MODEL} (GPU) — ~170k papers, "
          f"expect many minutes", flush=True)
    emb = db.generate_embeddings(
        source="arxiv_scale_papers", model=EMBED_MODEL,
        columns=["title", "abstract"], key="paper_id",
    )
    ids, vecs = _read_vectors(db, emb)
    dim = int(vecs.shape[1])
    print(f"[embed] pulled {len(ids)} vectors × {dim}-dim", flush=True)
    if len(ids) != len(paper_ids):
        # Every embeddable paper must come back; a mismatch means the embed pass
        # dropped rows and the corpus/id-list would disagree. Fail loudly.
        raise RuntimeError(
            f"embedded vector count {len(ids)} != corpus size {len(paper_ids)} — "
            f"the embed pass dropped rows; STOP and report.")

    # ---- the HELD-OUT split: last QUERY_COUNT by sorted paper_id are queries ----
    order = np.argsort(ids)  # stable lexical sort of the string paper_ids
    ids_sorted = [ids[i] for i in order]
    vecs_sorted = vecs[order]
    q = QUERY_COUNT
    corpus_ids = ids_sorted[:-q]
    corpus_vecs = vecs_sorted[:-q]
    query_ids = ids_sorted[-q:]
    query_vecs = vecs_sorted[-q:]
    # The split must be disjoint by construction; assert it so a future edit that
    # broke the partition (queries leaking into the corpus) fails here, not as a
    # silently inflated recall@1.
    if set(corpus_ids) & set(query_ids):
        raise RuntimeError("held-out split is not disjoint — queries leaked into the corpus")
    print(f"[split] corpus {len(corpus_ids)}  held-out queries {len(query_ids)}  "
          f"(disjoint by sorted-paper_id partition)", flush=True)

    # ---- committed id lists + vectors ----
    (IDS_DIR / "arxiv_scale.txt").write_text("\n".join(ids_sorted) + "\n")
    (ARTIFACTS / "scale_queries.txt").write_text("\n".join(query_ids) + "\n")
    _write_vectors(ARTIFACTS / "arxiv_vectors.parquet", corpus_ids, corpus_vecs)
    _write_vectors(ARTIFACTS / "scale_query_vectors.parquet", query_ids, query_vecs)

    print("\nemitted vector cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)
    print(f"  data/ids/arxiv_scale.txt  "
          f"({(IDS_DIR / 'arxiv_scale.txt').stat().st_size} bytes)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="grpc://127.0.0.1:50051",
                    help="connect() target — grpc://host:port for the GPU server.")
    args = ap.parse_args()
    db = jammi_ai.connect(args.target)
    emit(db)


if __name__ == "__main__":
    main()
