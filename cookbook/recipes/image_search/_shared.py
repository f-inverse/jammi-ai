"""Shared paths / helpers for the numbered image-search recipe scripts.

`example.py` is the single-process walkthrough wired into CI. The numbered
scripts (`01`..`04`) are a stepwise version for readers and run in sequence,
sharing a persistent artifact dir + the same model so each step builds on the
previous one's state. This module holds the bits they have in common.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURES = REPO_ROOT / "cookbook" / "fixtures"
IMAGE_CORPUS_DIR = FIXTURES / "tiny_image_corpus"
GOLDEN_PATH = FIXTURES / "tiny_image_golden.json"

# Default to the hermetic local fixture (offline CI). Override with
# JAMMI_IMAGE_MODEL=patentclip/PatentCLIP_Vit_B for the federal use case.
DEFAULT_MODEL = f"local:{FIXTURES / 'tiny_open_clip'}"
MODEL = os.environ.get("JAMMI_IMAGE_MODEL", DEFAULT_MODEL)

# Persistent scratch dir shared across the numbered steps in one sequence.
WORKDIR = Path(os.environ.get("JAMMI_IMAGE_WORKDIR", Path(tempfile.gettempdir()) / "jammi-image-search"))
ARTIFACT_DIR = WORKDIR / "artifacts"
CORPUS_PARQUET = WORKDIR / "corpus.parquet"
GOLDEN_PARQUET = WORKDIR / "golden.parquet"


def ensure_source(db, name: str, url: str, fmt: str = "parquet") -> None:
    """Register a source, tolerating a prior registration.

    The numbered scripts each connect to the same persistent artifact dir, so
    a source registered by an earlier step is still in the catalog. Re-adding
    it raises "already registered" — which is fine here, the URL is identical.
    `example.py` uses a fresh temp dir per run and does not need this.
    """
    try:
        db.add_source(name, url=url, format=fmt)
    except RuntimeError as err:
        if "already registered" not in str(err):
            raise


def load_corpus_table() -> pa.Table:
    """Every `img_*.png` under the corpus dir as (image_id, image-bytes)."""
    rows = sorted(IMAGE_CORPUS_DIR.glob("img_*.png"))
    assert rows, f"no corpus images under {IMAGE_CORPUS_DIR}"
    return pa.table(
        {
            "image_id": pa.array([p.stem for p in rows], type=pa.utf8()),
            "image": pa.array([p.read_bytes() for p in rows], type=pa.binary()),
        }
    )


def build_image_golden_table() -> pa.Table:
    """Flatten `tiny_image_golden.json` into (query_id, query_image, relevant_id).

    The `query_image` binary column is what puts `db.eval_embeddings` into
    image-query mode.
    """
    queries = json.loads(GOLDEN_PATH.read_text())
    query_ids: list[str] = []
    query_images: list[bytes] = []
    relevant_ids: list[str] = []
    for q in queries:
        image_bytes = (IMAGE_CORPUS_DIR / q["query_image"]).read_bytes()
        for rid in q["relevant_ids"]:
            query_ids.append(q["query_id"])
            query_images.append(image_bytes)
            relevant_ids.append(str(rid))
    return pa.table(
        {
            "query_id": pa.array(query_ids, type=pa.utf8()),
            "query_image": pa.array(query_images, type=pa.binary()),
            "relevant_id": pa.array(relevant_ids, type=pa.utf8()),
        }
    )
