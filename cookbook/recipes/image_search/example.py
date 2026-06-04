"""Image-to-image search over a tiny synthetic corpus with an OpenCLIP model.

End-to-end walkthrough: load a small image corpus -> generate vision
embeddings -> run cosine nearest-neighbour search with an image query ->
evaluate retrieval quality (Recall@K / MRR) against a held-out golden set.

The numbered scripts (`01-load-corpus.py` ... `04-eval.py`) decompose this
same flow step by step; this file runs all four phases in one process and is
the version wired into `tests/cookbook_smoke.py`.

Model. The default model is the hermetic `tiny_open_clip` fixture so the
recipe runs offline in CI in well under 60s. The federal use case driving
this recipe is PatentCLIP — to run against it, set:

    JAMMI_IMAGE_MODEL=patentclip/PatentCLIP_Vit_B

`patentclip/PatentCLIP_Vit_B` is downloaded from the Hugging Face Hub on
first use and produces 512-dim L2-normalized embeddings. Any OpenCLIP-format
vision model works the same way (OpenAI CLIP, LAION ViT-B/32, ...).

Run with `python cookbook/recipes/image_search/example.py`. Exits 0 on success.
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
IMAGE_CORPUS_DIR = FIXTURES / "tiny_image_corpus"
GOLDEN_PATH = FIXTURES / "tiny_image_golden.json"

# Default to the hermetic local fixture so CI runs offline. Override with
# JAMMI_IMAGE_MODEL=patentclip/PatentCLIP_Vit_B (the federal use case) or any
# other OpenCLIP-format model ID / `local:<path>`.
DEFAULT_MODEL = f"local:{FIXTURES / 'tiny_open_clip'}"
MODEL = os.environ.get("JAMMI_IMAGE_MODEL", DEFAULT_MODEL)


def load_corpus_table() -> pa.Table:
    """Read every `img_*.png` in the corpus dir into an Arrow table with the
    inline image bytes the embedding pipeline consumes.

    Schema: `image_id` (utf8 key), `image` (binary, the raw PNG bytes).
    """
    rows = sorted(IMAGE_CORPUS_DIR.glob("img_*.png"))
    assert rows, f"no corpus images under {IMAGE_CORPUS_DIR}"
    ids = [p.stem for p in rows]
    blobs = [p.read_bytes() for p in rows]
    return pa.table(
        {
            "image_id": pa.array(ids, type=pa.utf8()),
            "image": pa.array(blobs, type=pa.binary()),
        }
    )


def build_image_golden(json_path: Path) -> pa.Table:
    """Flatten the per-query golden JSON into the (query_id, query_image,
    relevant_id) shape `db.eval_embeddings` consumes in image mode.

    The presence of a `query_image` (binary) column is what switches the eval
    runner from text-query to image-query encoding.
    """
    queries = json.loads(json_path.read_text())
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


def main() -> int:
    print(f"image_search: model = {MODEL}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db = jammi_ai.connect(f"file://{str(tmp_path)}")

        # 1. Load the corpus images into a Parquet source (inline image bytes).
        corpus_parquet = tmp_path / "corpus.parquet"
        pq.write_table(load_corpus_table(), corpus_parquet)
        db.add_source("corpus", url=str(corpus_parquet), format="parquet")

        # 2. Generate vision embeddings over the `image` column. The model is
        #    auto-detected from its OpenCLIP config; output is L2-normalized.
        db.generate_embeddings(
            source="corpus",
            model=MODEL,
            columns=["image"],
            key="image_id",
            modality="image",
        )

        # 3. Encode a single image query and run cosine ANN search.
        query_png = (IMAGE_CORPUS_DIR / "queries" / "q_circle.png").read_bytes()
        query_vec = db.encode_query(model=MODEL, query=query_png, modality="image")
        assert query_vec, "query embedding must be non-empty"
        print(f"query embedding dim: {len(query_vec)}")

        results = db.search("corpus", query=query_vec, k=5)  # pyarrow.Table
        assert results.num_rows > 0, "search must return a non-empty top-K"
        top_ids = results.column("image_id").to_pylist()
        print(f"top-{results.num_rows} for q_circle: {top_ids}")

        # 4. Evaluate retrieval quality against the held-out golden set. The
        #    eval encodes each golden `query_image`, searches, and reports
        #    Recall@K / MRR per query and in aggregate. We measure and report
        #    — we do NOT assert a quality target (the fixture model has random
        #    weights; real numbers come from a real model like PatentCLIP).
        golden_parquet = tmp_path / "golden.parquet"
        pq.write_table(build_image_golden(GOLDEN_PATH), golden_parquet)
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

    print("image_search: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
