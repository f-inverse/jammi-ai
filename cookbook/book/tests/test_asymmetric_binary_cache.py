"""Cache-backed checks on the committed asymmetric-binary-threshold vertical.

These run on CPU against the committed cache (no engine build, no GPU) and
assert the vertical's load-bearing invariants: the committed corpus/query
matrices are real (right shape, dimension, disjoint held-out split), the
manifest's provenance is well-formed, and the checksums cover every committed
file. The anisotropy diagnosis and the recall numbers themselves are
recomputed LIVE by ``asymmetric-binary.qmd`` — this module never re-derives
them, only the committed cache's own shape and internal consistency.
"""

from __future__ import annotations

import hashlib

import pytest

from jammi_cookbook import contracts

_DIR = contracts._dataset_dir("asymmetric_binary")
_HAVE_CACHE = (_DIR / "manifest.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="asymmetric_binary cache not emitted")


@_needs_cache
def test_manifest_records_a_real_modernbert_source_with_provenance():
    manifest = contracts.load_artifact("asymmetric_binary.manifest")
    assert manifest["base_model"] == "answerdotai/ModernBERT-base"
    assert manifest["dim"] == 768
    assert manifest["corpus_rows"] > 0
    assert manifest["query_rows"] > 0
    source = manifest["source"]
    assert source["ref"], "manifest must record which git ref the vectors came from"
    assert source["commit"], "manifest must record the exact source commit"
    assert set(source["lfs_oids"]) == {"corpus_vectors.parquet", "query_vectors.parquet"}


@_needs_cache
def test_committed_vectors_match_manifest_shape_and_are_held_out():
    manifest = contracts.load_artifact("asymmetric_binary.manifest")
    corpus = contracts.load_artifact("asymmetric_binary.corpus_vectors")
    query = contracts.load_artifact("asymmetric_binary.query_vectors")

    assert corpus.num_rows == manifest["corpus_rows"]
    assert query.num_rows == manifest["query_rows"]
    assert corpus.column("vector").type.list_size == manifest["dim"]
    assert query.column("vector").type.list_size == manifest["dim"]

    corpus_ids = set(corpus.column("_row_id").to_pylist())
    query_ids = set(query.column("_row_id").to_pylist())
    assert not (corpus_ids & query_ids), "held-out split must be disjoint by construction"
    assert len(corpus_ids) == corpus.num_rows, "_row_id must be unique in the corpus"
    assert len(query_ids) == query.num_rows, "_row_id must be unique in the query set"


@_needs_cache
def test_checksums_cover_every_committed_file_and_match():
    checksums = contracts.load_artifact("asymmetric_binary.checksums")
    for name in ("corpus_vectors.parquet", "query_vectors.parquet", "manifest.json"):
        assert name in checksums, name
        digest = hashlib.sha256((_DIR / name).read_bytes()).hexdigest()[:16]
        assert digest == checksums[name], f"{name}: on-disk file does not match committed checksum"
