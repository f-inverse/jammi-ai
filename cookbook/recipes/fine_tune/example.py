"""Fine-tune `tiny_bert` with LoRA, then encode a query with the result.

Run with `python cookbook/recipes/fine_tune/example.py`. Exits 0 on
success. Slow on CPU (~30s) — excluded from the default smoke matrix;
gated behind `JAMMI_COOKBOOK_SLOW=1` in `tests/cookbook_smoke.py`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jammi_ai

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURES = REPO_ROOT / "cookbook" / "fixtures"
PAIRS_PATH = FIXTURES / "tiny_pairs.csv"
BASE_MODEL = f"local:{FIXTURES / 'tiny_bert'}"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db = jammi_ai.connect(f"file://{tmp}")

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
        model_id = job.model_id
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

    print("fine_tune: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
