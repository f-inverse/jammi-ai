"""Cache-backed checks on the committed scale-tier ANN-vs-exact cross-check (H2).

These run on CPU against the committed scale cache (no GPU, no re-embedding) and assert
the tier's load-bearing invariants: the cache files checksum-verify against their committed
``sha256[:16]``; the held-out query set is disjoint from the indexed corpus (so recall@1 is
a real floor, not a structural ≈1.0); the frozen ``usearch`` index loads and covers every
corpus row; and — the measured verdict — the recomputed ANN-vs-exact recall@k clears every
committed floor at k ∈ {1, 10, 100}.

The recall here is recomputed from the SAME committed vectors + frozen index the chapter
folds (an exact numpy cosine-kNN oracle vs the loaded ``usearch`` HNSW), so the test is a
true second implementation of the chapter's verdict, not a re-read of a recorded number.

If the LFS-backed cache is absent the heavy artifacts are skipped, but the committed golden
floors, once present, are always asserted reachable.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from jammi_cookbook import contracts

_SD = contracts._dataset_dir("scale")
# The vectors are Git-LFS pointers until materialized; a tiny pointer file is < 1 KiB,
# the real corpus parquet is hundreds of MiB. Gate the heavy checks on LFS being checked out.
_CORPUS = _SD / "arxiv_vectors.parquet"
_HAVE_CACHE = _CORPUS.exists() and _CORPUS.stat().st_size > 1_000_000
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="scale LFS cache not materialized")

_KS = (1, 10, 100)


def _load_rowmap(path):
    data = path.read_bytes()
    offset = 4  # u32 version header
    ids = []
    while offset < len(data):
        (length,) = struct.unpack_from("<I", data, offset)
        offset += 4
        ids.append(data[offset:offset + length].decode())
        offset += length
    return ids


def _vectors(name):
    table = contracts.load_artifact(name)
    ids = [str(x) for x in table.column("_row_id").to_pylist()]
    vecs = np.asarray(table.column("vector").to_pylist(), dtype=np.float32)
    return ids, vecs


def test_recall_floors_are_committed_and_reachable():
    """Every recall floor the chapter asserts exists in the committed golden metrics."""
    for k in _KS:
        rf = contracts.recall_floor(f"scale.recall_at_{k}")
        assert 0.0 < rf.floor <= rf.measured <= 1.0, f"recall@{k} floor/measured out of range"
        # floor is (measured - margin), one-sided, to a rounding tolerance.
        gap = abs((rf.measured - rf.margin) - rf.floor)
        assert gap < 1e-6, f"recall@{k}: floor must equal measured minus margin"


@_needs_cache
def test_cache_checksums_verify():
    """Every committed scale artifact matches its frozen sha256[:16] — no silent LFS drift."""
    checksums = contracts.load_artifact("scale.checksums")
    for name, expected in checksums.items():
        digest = hashlib.sha256((_SD / name).read_bytes()).hexdigest()[:16]
        assert digest == expected, f"{name}: checksum {digest} != committed {expected}"


@_needs_cache
def test_queries_are_held_out_of_the_corpus():
    """The recall query set is disjoint from the indexed corpus (held-out guarantee)."""
    corpus_ids, _ = _vectors("scale.corpus_vectors")
    query_ids, _ = _vectors("scale.query_vectors")
    assert not (set(corpus_ids) & set(query_ids)), "queries must not appear in the corpus"
    # the committed id list matches the query parquet
    committed = contracts.load_artifact("scale.queries")
    assert set(committed) == set(query_ids), "scale_queries.txt must match the query vectors"


@_needs_cache
def test_frozen_index_loads_and_covers_corpus():
    """The committed usearch index loads (view, no rebuild) and covers every corpus row."""
    from usearch.index import Index

    corpus_ids, corpus = _vectors("scale.corpus_vectors")
    index = Index.restore(str(contracts.load_artifact("scale.ann_index")), view=True)
    assert len(index) == len(corpus_ids), "frozen index must cover every corpus row"
    assert index.ndim == corpus.shape[1], "frozen index dim must match the corpus"
    rowmap = _load_rowmap(contracts.load_artifact("scale.ann_rowmap"))
    assert len(rowmap) == len(corpus_ids), "rowmap must cover every corpus row"


@_needs_cache
def test_recomputed_recall_clears_every_floor():
    """The measured verdict: recomputed ANN-vs-exact recall@k clears every committed floor.

    Exact numpy cosine-kNN oracle vs the loaded (never rebuilt) usearch HNSW, over the
    held-out queries — a second implementation of the chapter's recall, asserted >= floor.
    """
    from usearch.index import Index

    corpus_ids, corpus = _vectors("scale.corpus_vectors")
    _, queries = _vectors("scale.query_vectors")
    k_max = max(_KS)

    corpus_unit = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12)
    query_unit = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
    sims = query_unit @ corpus_unit.T
    topk = np.argpartition(-sims, k_max, axis=1)[:, :k_max]
    rows = np.arange(queries.shape[0])[:, None]
    order = np.argsort(-sims[rows, topk], axis=1)
    exact_idx = topk[rows, order]
    exact = [[corpus_ids[j] for j in exact_idx[i]] for i in range(queries.shape[0])]

    index = Index.restore(str(contracts.load_artifact("scale.ann_index")), view=True)
    rowmap = _load_rowmap(contracts.load_artifact("scale.ann_rowmap"))
    matches = index.search(queries, k_max)
    ann = [[rowmap[k] for k in matches.keys[i]] for i in range(queries.shape[0])]

    for k in _KS:
        observed = float(np.mean([
            len(set(exact[i][:k]) & set(ann[i][:k])) / k for i in range(queries.shape[0])
        ]))
        # asserts observed >= floor; raises with provenance if a real regression is caught.
        contracts.assert_recall_floor(f"scale.recall_at_{k}", observed)
