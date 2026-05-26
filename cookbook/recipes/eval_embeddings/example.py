"""Measure recall@k / nDCG of a vector index against a golden relevance set.

Run with `python cookbook/recipes/eval_embeddings/example.py`. Exits 0 on
success.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import jammi_ai

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURES = REPO_ROOT / "cookbook" / "fixtures"
CORPUS_PATH = FIXTURES / "tiny_corpus.parquet"
GOLDEN_PATH = FIXTURES / "tiny_golden.json"
MODEL = f"local:{FIXTURES / 'tiny_bert'}"


def expand_golden_to_csv(json_path: Path, out_path: Path) -> None:
    """Flatten the per-query-with-list-of-relevant-ids JSON into the
    one-row-per-(query, relevant_id) CSV that `eval_embeddings` ingests.
    """
    queries = json.loads(json_path.read_text())
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query_text", "relevant_id"])
        for q in queries:
            for rid in q["relevant_ids"]:
                writer.writerow([q["query_id"], q["query_text"], str(rid)])


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db = jammi_ai.connect(
            artifact_dir=str(tmp_path),
            gpu_device=-1,
            inference_batch_size=8,
        )

        # 1. Register the corpus and build the embedding index.
        db.add_source("corpus", url=str(CORPUS_PATH), format="parquet")
        db.generate_text_embeddings(
            source="corpus",
            model=MODEL,
            columns=["content"],
            key="id",
        )

        # 2. Expand the JSON golden set into the CSV shape eval_embeddings
        #    consumes, then register it as a source.
        golden_csv = tmp_path / "golden.csv"
        expand_golden_to_csv(GOLDEN_PATH, golden_csv)
        db.add_source("golden", url=str(golden_csv), format="csv")

        # 3. Run the retrieval eval.
        metrics = db.eval_embeddings(
            source="corpus",
            golden_source="golden.public.golden",
            k=5,
        )

        # 4. Sanity-check every aggregate metric is in [0, 1].
        print("aggregate:")
        for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
            value = metrics[key]
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"
            print(f"  {key:<16} {value:.4f}")

    print("eval_embeddings: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
