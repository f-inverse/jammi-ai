"""5-minute Jammi AI quickstart — register, embed, search.

Walks the four steps from `cookbook/quickstart/`'s README:

1. `jammi_ai.connect` — open a local session
2. `db.add_source` — attach the tiny corpus fixture
3. `db.generate_text_embeddings` — build a 32-dim USEARCH-backed index
4. `db.encode_text_query` + `db.search(...).run()` — execute a similarity query

Uses the local `cookbook/fixtures/tiny_bert` encoder so the script runs
without network access. Swap `MODEL` for a Hugging Face Hub model ID like
`sentence-transformers/all-MiniLM-L6-v2` for production.

Exits 0 in well under 30 seconds on CPU.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jammi_ai

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO_ROOT / "cookbook" / "fixtures"
CORPUS_PATH = FIXTURES / "tiny_corpus.parquet"
MODEL = f"local:{FIXTURES / 'tiny_bert'}"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        # 1. Connect — CPU-only so the example is reproducible.
        db = jammi_ai.connect(
            artifact_dir=tmp,
            gpu_device=-1,
            inference_batch_size=8,
        )

        # 2. Register the tiny corpus as a Parquet source.
        db.add_source("corpus", url=str(CORPUS_PATH), format="parquet")

        # 3. Build a 32-dim embedding table over the `content` column.
        db.generate_text_embeddings(
            source="corpus",
            model=MODEL,
            columns=["content"],
            key="id",
        )

        # 4. Encode a query and run a top-3 similarity search.
        query_vec = db.encode_text_query(MODEL, "how does quantum computing work?")
        results = db.search("corpus", query=query_vec, k=3).run()

        rows = results.to_pylist()
        if not rows:
            raise RuntimeError("quickstart returned zero rows")

        print("id        similarity  title")
        for row in rows:
            print(
                f"{row['_row_id']:<8}  {row['similarity']:>9.4f}  {row['title']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
