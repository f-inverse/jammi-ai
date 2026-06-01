"""Audio-to-audio search over a tiny synthetic corpus with a CLAP model.

End-to-end walkthrough: load a small audio corpus -> generate audio
embeddings -> run cosine nearest-neighbour search with an audio query ->
evaluate retrieval quality (Recall@K / MRR) against a held-out golden set.

The numbered scripts (`01-load-corpus.py` ... `04-eval.py`) decompose this
same flow step by step; this file runs all four phases in one process and is
the version wired into `tests/cookbook_smoke.py`.

Model. The default model is the hermetic `tiny_clap` fixture so the recipe
runs offline in CI in well under 60s. Any CLAP-format audio model works the
same way — point `JAMMI_AUDIO_MODEL` at a Hugging Face repo id or
`local:<path>` whose `open_clip_config.json` carries a `model_cfg.audio_cfg`
block and whose checkpoint exposes the `audio.*` tower keys:

    JAMMI_AUDIO_MODEL=laion/clap-htsat-unfused   # (illustrative)

The fixture has random weights, so its embeddings are garbage and the eval
numbers are not meaningful — the recipe measures and reports them to exercise
the full path; real retrieval quality comes from a real CLAP checkpoint.

Run with `python cookbook/recipes/audio_search/example.py`. Exits 0 on success.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import jammi_ai

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURES = REPO_ROOT / "cookbook" / "fixtures"
AUDIO_CORPUS_DIR = FIXTURES / "tiny_audio_corpus"
GOLDEN_PATH = FIXTURES / "tiny_audio_golden.json"

# Default to the hermetic local fixture so CI runs offline. Override with
# JAMMI_AUDIO_MODEL=<hf-repo-id> or `local:<path>` for any CLAP-format model.
DEFAULT_MODEL = f"local:{FIXTURES / 'tiny_clap'}"
MODEL = os.environ.get("JAMMI_AUDIO_MODEL", DEFAULT_MODEL)


def load_corpus_table() -> pa.Table:
    """Read every `clip_*.wav` in the corpus dir into an Arrow table with the
    inline audio bytes the embedding pipeline consumes.

    Schema: `clip_id` (utf8 key), `audio` (binary, the raw WAV bytes).
    """
    rows = sorted(AUDIO_CORPUS_DIR.glob("clip_*.wav"))
    assert rows, f"no corpus clips under {AUDIO_CORPUS_DIR}"
    ids = [p.stem for p in rows]
    blobs = [p.read_bytes() for p in rows]
    return pa.table(
        {
            "clip_id": pa.array(ids, type=pa.utf8()),
            "audio": pa.array(blobs, type=pa.binary()),
        }
    )


def build_audio_golden(json_path: Path) -> pa.Table:
    """Flatten the per-query golden JSON into the (query_id, query_audio,
    relevant_id) shape `db.eval_embeddings` consumes in audio mode.

    The presence of a `query_audio` (binary) column is what switches the eval
    runner from text/image-query to audio-query encoding.
    """
    queries = json.loads(json_path.read_text())
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


def main() -> int:
    print(f"audio_search: model = {MODEL}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db = jammi_ai.connect(
            artifact_dir=str(tmp_path),
            gpu_device=-1,
            inference_batch_size=8,
        )

        # 1. Load the corpus clips into a Parquet source (inline audio bytes).
        corpus_parquet = tmp_path / "corpus.parquet"
        pq.write_table(load_corpus_table(), corpus_parquet)
        db.add_source("corpus", url=str(corpus_parquet), format="parquet")

        # 2. Generate audio embeddings over the `audio` column. The model is
        #    auto-detected from its CLAP config; the backend owns
        #    decode -> resample -> log-mel -> forward; output is L2-normalized.
        db.generate_audio_embeddings(
            source="corpus",
            model=MODEL,
            audio_column="audio",
            key="clip_id",
        )

        # 3. Encode a single audio query and run cosine ANN search.
        query_wav = (AUDIO_CORPUS_DIR / "queries" / "q_sine.wav").read_bytes()
        query_vec = db.encode_audio_query(MODEL, query_wav)
        assert query_vec, "query embedding must be non-empty"
        print(f"query embedding dim: {len(query_vec)}")

        results = db.search("corpus", query=query_vec, k=5).run()
        assert results.num_rows > 0, "search must return a non-empty top-K"
        top_ids = results.column("clip_id").to_pylist()
        print(f"top-{results.num_rows} for q_sine: {top_ids}")

        # 4. Evaluate retrieval quality against the held-out golden set. The
        #    eval encodes each golden `query_audio`, searches, and reports
        #    Recall@K / MRR per query and in aggregate. We measure and report
        #    — we do NOT assert a quality target (the fixture model has random
        #    weights; real numbers come from a real CLAP checkpoint).
        golden_parquet = tmp_path / "golden.parquet"
        pq.write_table(build_audio_golden(GOLDEN_PATH), golden_parquet)
        db.add_source("golden", url=str(golden_parquet), format="parquet")

        metrics = db.eval_embeddings(
            source="corpus",
            golden_source="golden.public.golden",
            k=5,
        )

        aggregate = metrics["aggregate"]
        print("aggregate retrieval metrics:")
        for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
            value = aggregate[key]
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"
            print(f"  {key:<16} {value:.4f}")

        per_query = metrics["per_query"]
        assert len(per_query) > 0, "per_query must carry one record per query"
        print(f"per_query: {len(per_query)} records")

    print("audio_search: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
