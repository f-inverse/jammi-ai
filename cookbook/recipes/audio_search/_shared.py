"""Shared paths / helpers for the numbered audio-search recipe scripts.

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
AUDIO_CORPUS_DIR = FIXTURES / "tiny_audio_corpus"
GOLDEN_PATH = FIXTURES / "tiny_audio_golden.json"

# Default to the hermetic local fixture (offline CI). Override with
# JAMMI_AUDIO_MODEL=<hf-repo-id> or `local:<path>` for any CLAP-format model.
DEFAULT_MODEL = f"local:{FIXTURES / 'tiny_clap'}"
MODEL = os.environ.get("JAMMI_AUDIO_MODEL", DEFAULT_MODEL)

# Persistent scratch dir shared across the numbered steps in one sequence.
WORKDIR = Path(
    os.environ.get("JAMMI_AUDIO_WORKDIR", Path(tempfile.gettempdir()) / "jammi-audio-search")
)
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
    """Every `clip_*.wav` under the corpus dir as (clip_id, audio-bytes)."""
    rows = sorted(AUDIO_CORPUS_DIR.glob("clip_*.wav"))
    assert rows, f"no corpus clips under {AUDIO_CORPUS_DIR}"
    return pa.table(
        {
            "clip_id": pa.array([p.stem for p in rows], type=pa.utf8()),
            "audio": pa.array([p.read_bytes() for p in rows], type=pa.binary()),
        }
    )


def build_audio_golden_table() -> pa.Table:
    """Flatten `tiny_audio_golden.json` into (query_id, query_audio, relevant_id).

    The `query_audio` binary column is what puts `db.eval_embeddings` into
    audio-query mode.
    """
    queries = json.loads(GOLDEN_PATH.read_text())
    query_ids: list[str] = []
    query_audios: list[bytes] = []
    relevant_ids: list[str] = []
    for q in queries:
        audio_bytes = (AUDIO_CORPUS_DIR / q["query_audio"]).read_bytes()
        for rid in q["relevant_ids"]:
            query_ids.append(q["query_id"])
            query_audios.append(audio_bytes)
            relevant_ids.append(str(rid))
    return pa.table(
        {
            "query_id": pa.array(query_ids, type=pa.utf8()),
            "query_audio": pa.array(query_audios, type=pa.binary()),
            "relevant_id": pa.array(relevant_ids, type=pa.utf8()),
        }
    )
