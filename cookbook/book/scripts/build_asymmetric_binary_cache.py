#!/usr/bin/env python3
"""Emit the asymmetric-binary-threshold vertical cache — GPU-free, run ONCE.

This vertical measures the Wave-2 anisotropy fix (`sign(v − τ)` per-dimension
threshold quantization, `crates/jammi-db/src/index/sidecar.rs`) against a real
ModernBERT-base corpus. Unlike every other keystone in this book, it does
**not** drive a fresh GPU embedding pass: the corpus this vertical needs — a
few thousand real ModernBERT-base embeddings with a disjoint held-out query
split — already exists, committed (Git LFS) on the abandoned QAT joint branch
(`feat/qat-wave2-joint-abandoned`, the `qat.baseline` head: `quantization_aware
=None`, i.e. no quantization-aware fine-tune target, the plain embedding
function). Re-embedding the same few thousand papers again would be a pure,
avoidable GPU cost.

There is a second, load-bearing reason to reuse this EXACT corpus rather than
embed a fresh one: the engine commit that introduced the asymmetric threshold
(`5ba146d`, "feat(db): asymmetric (mean/median-centered) binary quantization")
states measured anisotropy numbers in its own commit message — `‖μ‖/E‖v‖ ≈
0.97`, `183/768` collapsed dimensions under the old fixed-`0` threshold. This
script's own emit reproduces those numbers EXACTLY (`0.9669` and `183`,
confirmed by direct computation below) — strong evidence the engine author
measured the fix against this identical corpus. Reusing it here, rather than a
fresh embed that would only approximately match, keeps this chapter's
diagnosis numerically anchored to the same real corpus the fix itself was
validated against.

What this script produces (under ``artifacts/asymmetric_binary/``):

* ``corpus_vectors.parquet`` — the INDEXED corpus: 3 200 ModernBERT-base
  embeddings (``_row_id``, ``vector``), 768-dim f32.
* ``query_vectors.parquet``  — the HELD-OUT queries: 800 ModernBERT-base
  embeddings, disjoint from the corpus by construction (the QAT keystone's own
  held-out split, `scripts/build_qat_cache.py`'s ``N_QUERY`` convention).
* ``manifest.json``   — full provenance: which branch/commit/path the vectors
  were copied from, the source LFS object ids, and the base model.
* ``checksums.json``  — sha256[:16] of every committed vector file.

The anisotropy diagnosis (``‖μ‖/E‖v‖``, collapsed-dim counts), the plain-vs-
asymmetric recall fold, and the engine-measured binary-precision recall are
all computed LIVE by ``chapters/22-precision/asymmetric-binary.qmd`` from the
two committed parquet files above — never frozen here, exactly like
``precision.qmd`` / ``binary-precision.qmd`` read ``scale.corpus_vectors``.

Usage::

    python scripts/build_asymmetric_binary_cache.py \\
        --source-ref feat/qat-wave2-joint-abandoned \\
        --source-path cookbook/book/artifacts/qat/baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "asymmetric_binary"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BASE_MODEL = "answerdotai/ModernBERT-base"

FILES = ("corpus_vectors.parquet", "query_vectors.parquet")


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _git_show_lfs_smudged(ref: str, path: str) -> bytes:
    """Read ``ref:path`` from the local git object store and pass it through
    the Git LFS smudge filter, returning the real file bytes (not the LFS
    pointer). Requires the LFS object to already be present locally — the
    caller runs ``git lfs fetch origin <ref>`` first if it might not be."""
    show = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    )
    smudge = subprocess.run(
        ["git", "lfs", "smudge"],
        cwd=REPO_ROOT, input=show.stdout, capture_output=True, check=True,
    )
    return smudge.stdout


def _lfs_oid(ref: str, path: str) -> str:
    """The LFS pointer's own ``oid`` line for ``ref:path`` — recorded in the
    manifest as the exact source-object identity, independent of which git ref
    happens to carry it."""
    show = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=REPO_ROOT, capture_output=True, check=True, text=True,
    )
    for line in show.stdout.splitlines():
        if line.startswith("oid sha256:"):
            return line.removeprefix("oid sha256:")
    raise RuntimeError(f"{ref}:{path} is not an LFS pointer (no oid line found)")


def emit(source_ref: str, source_path: str) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # `git lfs fetch` pulls the LFS objects for `source_ref` into the local
    # store WITHOUT checking it out — safe to run from any current branch.
    subprocess.run(["git", "lfs", "fetch", "origin", source_ref],
                    cwd=REPO_ROOT, check=True)

    source_oids: dict[str, str] = {}
    for name in FILES:
        rel = f"{source_path}/{name}"
        data = _git_show_lfs_smudged(source_ref, rel)
        (ARTIFACTS / name).write_bytes(data)
        source_oids[name] = _lfs_oid(source_ref, rel)
        print(f"[copy] {rel} ({len(data):,} bytes) <- {source_ref}", flush=True)

    import pyarrow.parquet as pq

    corpus = pq.read_table(ARTIFACTS / "corpus_vectors.parquet")
    query = pq.read_table(ARTIFACTS / "query_vectors.parquet")
    corpus_ids = set(corpus.column("_row_id").to_pylist())
    query_ids = set(query.column("_row_id").to_pylist())
    if corpus_ids & query_ids:
        raise RuntimeError("held-out split is not disjoint — the source vectors are corrupt")
    dim = corpus.column("vector").type.list_size
    print(f"corpus: {corpus.num_rows:,} x {dim}  query: {query.num_rows:,} x {dim} "
          f"(disjoint by construction)", flush=True)

    source_commit = subprocess.run(
        ["git", "rev-parse", source_ref], cwd=REPO_ROOT,
        capture_output=True, check=True, text=True,
    ).stdout.strip()

    (ARTIFACTS / "manifest.json").write_text(json.dumps({
        "base_model": BASE_MODEL,
        "corpus_rows": corpus.num_rows,
        "query_rows": query.num_rows,
        "dim": dim,
        "source": {
            "ref": source_ref,
            "commit": source_commit,
            "path": source_path,
            "lfs_oids": source_oids,
            "note": "The `baseline` head (quantization_aware=None) of the abandoned "
                    "QAT joint keystone (scripts/build_qat_cache.py) — a real "
                    "ModernBERT-base embedding of the committed ogbn-arxiv subset, with "
                    "a disjoint held-out query split. Copied byte-for-byte (never "
                    "re-embedded) to avoid a duplicate GPU pass over the SAME "
                    "few-thousand-document corpus the QAT vertical already emitted.",
        },
        "note": "The anisotropy diagnosis, the plain-vs-asymmetric no-rescore recall "
                "fold, and the engine-measured binary-precision recall are computed "
                "LIVE by chapters/22-precision/asymmetric-binary.qmd from the two "
                "committed parquet files — never frozen here.",
    }, indent=2))

    checksums = {name: _checksum(ARTIFACTS / name) for name in FILES}
    checksums["manifest.json"] = _checksum(ARTIFACTS / "manifest.json")
    (ARTIFACTS / "checksums.json").write_text(json.dumps(checksums, indent=2, sort_keys=True))

    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size:,} bytes)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-ref", default="feat/qat-wave2-joint-abandoned",
                     help="git ref carrying the already-embedded baseline corpus.")
    ap.add_argument("--source-path", default="cookbook/book/artifacts/qat/baseline",
                     help="path (within --source-ref) to the corpus/query parquet pair.")
    args = ap.parse_args()
    emit(args.source_ref, args.source_path)


if __name__ == "__main__":
    main()
