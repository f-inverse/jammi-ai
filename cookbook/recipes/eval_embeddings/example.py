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
        db = jammi_ai.connect(f"file://{str(tmp_path)}")

        # 1. Register the corpus and build the embedding index.
        db.add_source("corpus", url=str(CORPUS_PATH), format="parquet")
        db.generate_embeddings(
            source="corpus",
            model=MODEL,
            columns=["content"],
            key="id",
            modality="text",
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
        aggregate = metrics["aggregate"]
        print("aggregate:")
        for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
            value = aggregate[key]
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"
            print(f"  {key:<16} {value:.4f}")

        # 5. Drill into the per-query records (one entry per golden-set query).
        per_query = metrics["per_query"]
        assert len(per_query) > 0, "per_query must carry one record per query"
        first = per_query[0]
        assert first["query_id"], "per_query records carry the golden-set query_id"
        print(f"per_query: {len(per_query)} records (first: {first['query_id']})")

        # 6. Per-query results are also persisted in the catalog, keyed by the
        #    run's eval_run_id. Read them back to inspect Recall@{1,3,5,10},
        #    MRR, nDCG, and distance for each query — and any cohort tags you
        #    attached at eval time (see step 7).
        eval_run_id = metrics["eval_run_id"]
        persisted = db.eval_per_query(eval_run_id)
        assert len(persisted) == len(per_query), "one persisted row per query"
        row = persisted[0]
        for key in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg", "distance"):
            assert key in row["metrics"], f"persisted metric '{key}' present"
        print(f"persisted per-query rows: {len(persisted)} (run {eval_run_id})")

        # 7. Optional: attach opaque cohort tags per query_id so you can later
        #    aggregate quality by segment. The substrate stores them verbatim;
        #    it never interprets the keys/values.
        tagged = db.eval_embeddings(
            source="corpus",
            golden_source="golden.public.golden",
            k=5,
            cohorts={"q1": {"split": "val"}},
        )
        tagged_rows = db.eval_per_query(tagged["eval_run_id"])
        by_query = {r["query_id"]: r for r in tagged_rows}
        if "q1" in by_query:
            assert by_query["q1"]["cohorts"].get("split") == "val", "cohort stored verbatim"
            print("cohort tag round-trip: OK (q1 -> {'split': 'val'})")

    print("eval_embeddings: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
