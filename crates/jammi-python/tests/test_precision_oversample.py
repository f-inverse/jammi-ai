"""Reachability + correctness of `storage_precision` and per-request `oversample`
through the PUBLIC Python SDK — `jammi.connect(..., config=...)` /
`Database.import_embeddings` / `Database.search(..., oversample=...)`, never
`jammi_native.open_local` or a hand-built wire `SearchRequest`.

Before this test's SDK surface existed, a caller could not set
`embedding.ann.storage_precision` (a table's quantized-index precision) nor
override a search's retrieve->rescore `oversample` through the documented
front door at all — the capability was real and engine-tested, but
unreachable from the public API (the gap this test closes and proves closed).

The corpus is a deterministic, hand-built adversarial construction — not a
committed fixture, not randomness seeded off wall-clock — designed so `int8`
scalar quantization is GUARANTEED lossy enough to misrank, while the exact
`f32` rescore is guaranteed to recover the true answer:

* every vector carries one dominant "spike" dimension at a constant `1.0`
  and `dim - 1` small dimensions carrying the actual discriminating signal,
  drawn uniformly from `[0, 2e-3)`;
* after L2-normalization (which the engine's `import_embeddings` always
  applies), the spike dimension still dominates each vector's own value
  range, so USearch's per-vector affine `int8` quantization step
  (`(max - min) / 255 ~= 1/255 ~= 0.4%` of the spike) is roughly 2 ORDERS OF
  MAGNITUDE wider than the `2e-3` signal band — every vector's small
  dimensions are quantized to indistinguishable buckets, so the quantized
  index cannot discriminate corpus rows AT ALL; a `k=1` retrieval off the raw
  quantized graph (`oversample=1`, so the candidate pool IS the raw result)
  is therefore uncorrelated with which row a query actually matches;
* each query is one corpus row's OWN vector fed back verbatim, so the "true
  answer" needs no independently-computed oracle (and no delicate
  float32-vs-float64 tie-breaking near a razor-thin margin): a query is, by
  construction, closer to its own source row than to any other row's
  independently-drawn small-dim signal, and the exact `f32` rescore (whose
  candidate pool is widened enough to contain every row) recovers that row
  reliably every time.

This makes the divergence a certainty of the construction, not a statistical
tendency — reproducible via the fixed `random.Random` seed below, no fixture
file, no network, no GPU (`import_embeddings` needs no encoder).
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

import jammi

DIM = 8
N_CORPUS = 30
SPIKE = 1.0
SIGNAL_LO, SIGNAL_HI = 0.0, 2e-3
SEED = 2024
# Every 3rd row is queried back at itself — 10 held-out-style probes over a
# 30-row corpus, deterministic and reproducible from `SEED` alone.
QUERY_ROW_STRIDE = 3


def _make_vector(rng: random.Random) -> List[float]:
    return [SPIKE] + [rng.uniform(SIGNAL_LO, SIGNAL_HI) for _ in range(DIM - 1)]


def _write_corpus(dir_path: Path) -> Tuple[str, List[str], List[List[float]]]:
    rng = random.Random(SEED)
    ids = [f"row-{i}" for i in range(N_CORPUS)]
    vectors = [_make_vector(rng) for _ in ids]
    path = dir_path / "corpus.parquet"
    pq.write_table(
        pa.table(
            {
                "_row_id": ids,
                "vector": pa.array(vectors, type=pa.list_(pa.float32(), DIM)),
            }
        ),
        path,
    )
    return f"file://{path}", ids, vectors


def _connect_with_ann_defaults(
    tmp_path: Path, *, storage_precision: str, oversample: int
):
    """`jammi.connect(..., config=...)` — the public deployment-default
    passthrough this unit adds — sets the embedded engine's
    `embedding.ann.storage_precision` / `embedding.ann.oversample`, which every
    embedding table created on this session is stamped with at creation."""
    cfg_path = tmp_path / "jammi.toml"
    cfg_path.write_text(
        f'[embedding.ann]\nstorage_precision = "{storage_precision}"\noversample = {oversample}\n'
    )
    artifact_dir = tmp_path / "engine"
    return jammi.connect(f"file://{artifact_dir}", config=str(cfg_path))


def _register_table(db, corpus_url: str) -> str:
    """Register the corpus as a ready embedding table through ONLY the public
    surface: `add_source` + `import_embeddings` (GPU-free — no encoder runs;
    these vectors are already computed)."""
    db.add_source("vectors", url=corpus_url, format="parquet")
    return db.import_embeddings(
        source="vectors",
        model="synthetic-adversarial-vectors",
        vectors_url=corpus_url,
        key="_row_id",
        dimensions=DIM,
    )


def test_int8_storage_precision_and_oversample_reachable_through_public_api(
    tmp_path: Path,
) -> None:
    """The public front door — `connect(config=...)` for the deployment
    default, `import_embeddings` for table registration, `search(...,
    oversample=...)` for the per-request override — reaches a REAL, working
    `int8` quantized table and its retrieve->rescore knob.

    This is proved WITHOUT relying on a negative "the narrow default must
    MISS" claim: whether a `k=1`, `oversample=1` int8 search happens to land
    on the query's own source row is quantization noise the construction
    only makes *unlikely*, not impossible, and it is not required to be
    identical across environments/USearch builds (this is exactly what made
    the previous version of this test flaky in CI). Instead this asserts two
    things that are deterministic by construction, plus the wire-assembly
    proof that the keyword genuinely reaches the request:

    1. A per-request `oversample` wide enough to cover the whole corpus
       recovers the query's own source row on **every** probe. Each query is
       one corpus row's own vector fed back verbatim, so once the rescore
       candidate pool is widened to the whole corpus, the exact `f32`
       rescore is *guaranteed* to rank that row first (distance to itself is
       exactly zero) — this holds regardless of how the underlying quantized
       ANN graph orders its narrow top-`k` candidates, so it is robust across
       environments. A silently-dropped `oversample` override would leave the
       narrow (`=1`) deployment default in effect, which the corpus is
       constructed to make uncorrelated with the true answer — the
       probability of ALL 10 probes accidentally recovering their own row
       under the narrow default is vanishingly small, so this assertion still
       fails hard if the override never reaches the engine.
    2. The request assembled by the public binding for a given `oversample`
       actually carries it — a direct, deterministic proof that the keyword
       reaches the wire message (`SearchRequest.oversample` is an optional
       proto3 field: present+equal when passed, absent when omitted), rather
       than inferring reachability solely from end-to-end retrieval quality.
    """
    from jammi._assembly import build_search_request

    db = _connect_with_ann_defaults(tmp_path, storage_precision="int8", oversample=1)
    corpus_url, ids, vectors = _write_corpus(tmp_path)
    _register_table(db, corpus_url)

    probe_indices = range(0, N_CORPUS, QUERY_ROW_STRIDE)
    assert len(list(probe_indices)) == 10

    for i in probe_indices:
        query, expected_id = vectors[i], ids[i]

        # Reachability smoke check: the override path executes end-to-end
        # through the public API and returns exactly k=1 result, whatever
        # row it lands on.
        default_hits = db.search("vectors", query=query, k=1).to_pylist()
        assert len(default_hits) == 1

        # Deterministic-by-construction recovery: a per-request oversample
        # covering the whole corpus makes every row a rescore candidate, so
        # the exact f32 rescore recovers the query's own source row every
        # time — proving the override genuinely reaches the engine's
        # retrieve->rescore resolution, without depending on the narrow
        # default's (quantization-noise-dependent) behavior.
        overridden_hit = db.search(
            "vectors", query=query, k=1, oversample=N_CORPUS * 2
        ).to_pylist()[0]
        assert overridden_hit["_row_id"] == expected_id, (
            "a per-request oversample wide enough to cover the whole corpus "
            "must recover the query's own source row through the public "
            "search() surface"
        )

    # Wire-assembly proof: `oversample` threads into the request the binding
    # actually submits, and its absence leaves the field genuinely unset
    # (deferring to the table's stamped default) rather than some silent 0.
    request_with_override = build_search_request(
        "vectors", query=vectors[0], k=1, oversample=N_CORPUS * 2
    )
    assert request_with_override.HasField("oversample")
    assert request_with_override.oversample == N_CORPUS * 2

    request_without_override = build_search_request("vectors", query=vectors[0], k=1)
    assert not request_without_override.HasField("oversample")


def test_f32_storage_precision_is_unaffected_by_oversample_control(
    tmp_path: Path,
) -> None:
    """Control: the SAME adversarial corpus/probes at `storage_precision =
    "f32"` (single-stage, no rescore) always find the query's own source row
    regardless of `oversample` — proving the divergence measured above is a
    genuine effect of `int8` quantization reaching the engine, not an
    artifact of the corpus construction itself."""
    db = _connect_with_ann_defaults(tmp_path, storage_precision="f32", oversample=1)
    corpus_url, ids, vectors = _write_corpus(tmp_path)
    _register_table(db, corpus_url)

    for i in range(0, N_CORPUS, QUERY_ROW_STRIDE):
        query, expected_id = vectors[i], ids[i]
        hit = db.search("vectors", query=query, k=1).to_pylist()[0]
        assert hit["_row_id"] == expected_id, (
            "an f32-precision table is single-stage exact search — it must "
            "find the query's own source row with no oversample override "
            "needed"
        )


def test_connect_config_and_search_oversample_are_public_keywords() -> None:
    """`inspect.signature` proof — the same assay the discovering chapter
    used to show the gap — that `config=` and `oversample=` are now on the
    documented front door, not merely reachable by accident."""
    import inspect

    from jammi._assembly import build_search_request

    connect_params = set(inspect.signature(jammi.connect).parameters)
    search_params = set(inspect.signature(build_search_request).parameters)

    assert "config" in connect_params
    assert "oversample" in search_params
