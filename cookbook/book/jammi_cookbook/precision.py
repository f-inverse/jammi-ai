"""The corpus and tables the storage-precision chapters measure.

The papers are embedded once, at ``f32``; every quantized table a chapter builds
imports those same vectors, so the only thing that varies between two tables is
their ``storage_precision``. Every tenth paper is a held-out query (capped per
scale), and the ground truth each recall is scored against is the engine's own
exact search over the ``f32`` table.
"""

from __future__ import annotations

import contextlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import jammi

from . import datasets, encoders
from .scale import Scale

SOURCE = "papers"
K = 10
_QUERIES = {Scale.SMALL: 40, Scale.FULL: 200}


@dataclass(frozen=True)
class Corpus:
    """The embedded papers, their held-out queries, and the exact top-``K``."""

    work: Path
    model: str
    dims: int
    corpus_ids: list[str]
    vectors: np.ndarray
    queries: np.ndarray
    exact: list[list[str]]

    @property
    def exact_top1(self) -> list[str]:
        return [row[0] for row in self.exact]


def session(work: Path, storage_precision: str, oversample: int | None, extra: str = ""):
    """A fresh engine whose tables are stamped at ``storage_precision`` — and at
    a table-default ``oversample`` when one is given; ``None`` leaves the key out,
    so the precision's own default applies. ``extra`` lines join
    ``[embedding.ann]``."""
    config = Path(tempfile.mkdtemp(dir=work)) / "jammi.toml"
    body = f'[embedding.ann]\nstorage_precision = "{storage_precision}"\n'
    if oversample is not None:
        body += f"oversample = {oversample}\n"
    config.write_text(body + extra)
    return jammi.connect(f"file://{tempfile.mkdtemp(dir=work)}", config=str(config))


def corpus(scale: Scale) -> Corpus:
    """Embed the papers once and hold out the queries."""
    work = Path(tempfile.mkdtemp())
    model = encoders.text(scale)
    with session(work, "f32", None) as db:
        arxiv = datasets.arxiv(db, scale)
        papers = db.sql(
            f"SELECT paper_id, title || '. ' || abstract AS text "
            f"FROM {arxiv.papers}.public.{arxiv.papers} ORDER BY paper_id"
        ).to_pylist()
        queries = papers[::10][: _QUERIES[scale]]
        held = {q["paper_id"] for q in queries}
        pq.write_table(pa.Table.from_pylist([p for p in papers if p["paper_id"] not in held]),
                       work / "corpus.parquet")
        db.add_source(SOURCE, url=str(work / "corpus.parquet"), format="parquet")
        table = db.generate_embeddings(source=SOURCE, model=model, columns=["text"],
                                       key="paper_id")
        query_vectors = np.asarray(
            [db.encode_query(model=model, query=q["text"]) for q in queries], dtype=np.float32)
        exact = [db.search(SOURCE, query=q.tolist(), k=K, exact=True).column("_row_id").to_pylist()
                 for q in query_vectors]
        vectors = db.sql(f'SELECT _row_id, vector FROM "jammi.{table}" ORDER BY _row_id')
    pq.write_table(vectors, work / "vectors.parquet")
    return Corpus(
        work=work,
        model=model,
        dims=query_vectors.shape[1],
        corpus_ids=[str(r) for r in vectors.column("_row_id").to_pylist()],
        vectors=np.asarray(vectors.column("vector").to_pylist(), dtype=np.float32),
        queries=query_vectors,
        exact=exact,
    )


@contextlib.contextmanager
def built(c: Corpus, storage_precision: str, oversample: int | None, extra: str = ""):
    """A session holding one table of ``c``'s vectors at ``storage_precision``;
    yields the session and the table's name, and closes the session after."""
    with session(c.work, storage_precision, oversample, extra) as db:
        db.add_source(SOURCE, url=str(c.work / "corpus.parquet"), format="parquet")
        table = db.import_embeddings(source=SOURCE, model=c.model,
                                     vectors_url=str(c.work / "vectors.parquet"),
                                     key="paper_id", dimensions=c.dims)
        yield db, table


def recall(db, c: Corpus, k: int = K, oversample: int | None = None) -> float:
    """Mean recall@``k`` of ``db``'s search against the exact top-``k``; at
    ``k = 1``, the fraction of queries whose best hit is the true nearest."""
    knob = {} if oversample is None else {"oversample": oversample}
    scores = []
    for q, truth in zip(c.queries, c.exact):
        got = db.search(SOURCE, query=q.tolist(), k=k, **knob).column("_row_id").to_pylist()
        scores.append(len(set(truth[:k]) & set(got)) / k)
    return float(np.mean(scores))


def bundle_files(db, table: str) -> dict[str, list[Path]]:
    """The index's sidecar files by extension, one per segment in segment
    order, found beside each segment's ``index_path`` as the engine reports it."""
    files: dict[str, list[Path]] = {}
    for segment in db.list_index_segments(table):
        base = Path(urlparse(segment["index_path"]).path)
        for f in sorted(base.parent.glob(f"{base.stem}.*")):
            files.setdefault(f.name[len(base.stem) + 1:], []).append(f)
    return files


def bundle_bytes(db, table: str) -> dict[str, int]:
    """The total byte size of the index's sidecar files, by extension."""
    return {ext: sum(f.stat().st_size for f in fs) for ext, fs in bundle_files(db, table).items()}
