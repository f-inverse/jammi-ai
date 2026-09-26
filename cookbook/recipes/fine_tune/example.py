"""Fine-tune `tiny_bert` with LoRA, then encode a query with the result.

Then fine-tune from a citation graph instead of labelled pairs.

Run with `python cookbook/recipes/fine_tune/example.py`. Exits 0 on
success; seconds on CPU.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jammi
from jammi_cookbook import fixtures

PAIRS_PATH = fixtures.path("tiny_pairs.csv")
BASE_MODEL = fixtures.model("tiny_bert")
CITATIONS = fixtures.path("tiny_citation_graph")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp, jammi.connect(f"file://{tmp}") as db:

        # 1. Register the contrastive training pairs.
        db.add_source("training", url=str(PAIRS_PATH), format="csv")

        # 2. Submit the fine-tune job. Defaults are tuned for production
        #    workloads; for the cookbook we keep rank small and run a
        #    single epoch so the example finishes quickly.
        job = db.fine_tune(
            source="training",
            base_model=BASE_MODEL,
            columns=["text_a", "text_b", "score"],
            method="lora",
            task="text_embedding",
            lora_rank=4,
            epochs=1,
        )
        assert job.job_id, "fine_tune returned a job without an id"
        print(f"job_id:    {job.job_id}")

        # 3. Block until the job reaches a terminal state.
        job.wait()

        # 4. Newly-registered model_id follows the jammi:fine-tuned:* shape.
        model_id = job.output_model_id
        assert model_id.startswith("jammi:fine-tuned:"), (
            f"unexpected model_id: {model_id}"
        )
        print(f"model_id:  {model_id}")

        # 5. Encode a query through the fine-tuned model to confirm it
        #    loads end-to-end from the catalog.
        query_vec = db.encode_query(model=model_id, query="quantum computing applications")
        assert len(query_vec) == 32, (
            f"tiny_bert is 32-dim; got {len(query_vec)}-dim from fine-tuned"
        )

        # 6. Fine-tune from a graph instead of labelled pairs: nodes that cite
        #    each other are pulled together. Random walks over the citation
        #    edges sample the positives, the graph's non-neighbours the
        #    negatives. The nodes and edges are two ordinary sources.
        db.add_source("papers", url=str(CITATIONS / "nodes.jsonl"), format="jsonl")
        db.add_source("cites", url=str(CITATIONS / "edges.jsonl"), format="jsonl")
        graph_job = db.fine_tune_graph(
            node_source="papers",
            id_column="id",
            text_column="text",
            edge_source="cites",
            base_model=BASE_MODEL,
            lora_rank=4,
            epochs=1,
            sample_seed=0,
        )
        graph_job.wait()
        print(f"graph-tuned model_id: {graph_job.output_model_id}")
        assert graph_job.output_model_id.startswith("jammi:fine-tuned:")

    print("fine_tune: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
