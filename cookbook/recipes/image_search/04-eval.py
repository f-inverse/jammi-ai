"""Step 4 — evaluate retrieval quality against the golden set.

Builds the image-query golden source from `tiny_image_golden.json`, then calls
`db.eval_embeddings`. With a `query_image` column present the eval encodes each
golden image, searches the corpus index, and computes Recall@K / precision@K /
MRR / nDCG per query and in aggregate.

This script MEASURES and REPORTS the numbers — it does not assert an accuracy
target. The shipped fixture model has random weights (its numbers are
meaningless); run with JAMMI_IMAGE_MODEL=patentclip/PatentCLIP_Vit_B to get
real retrieval quality.
"""

from __future__ import annotations

import pyarrow.parquet as pq

import jammi_ai

from _shared import (
    ARTIFACT_DIR,
    CORPUS_PARQUET,
    GOLDEN_PARQUET,
    build_image_golden_table,
    ensure_source,
)


def main() -> int:
    assert CORPUS_PARQUET.exists(), "run 01..03 first"

    db = jammi_ai.connect(f"file://{str(ARTIFACT_DIR)}")
    ensure_source(db, "corpus", str(CORPUS_PARQUET))

    pq.write_table(build_image_golden_table(), GOLDEN_PARQUET)
    ensure_source(db, "golden", str(GOLDEN_PARQUET))

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
    print(f"per_query: {len(per_query)} records")
    for record in per_query:
        print(f"  {record['query_id']}")

    print("04-eval: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
