"""Step 3 — search the index with an audio query.

Encodes a held-out query clip into a vector and runs cosine nearest-neighbour
search over the corpus index built in step 02. Prints the top-K clip IDs.

`db.search(...)` returns a `pyarrow.Table` directly, with `filter=` / `select=`
for the bounded knobs; compound retrieval (join, or running a model over the
results with `annotate(...)`) is `db.sql(...)`. Here we keep it to a plain top-K.
"""

from __future__ import annotations

import jammi_ai

from _shared import ARTIFACT_DIR, AUDIO_CORPUS_DIR, CORPUS_PARQUET, MODEL, ensure_source


def main() -> int:
    assert CORPUS_PARQUET.exists(), "run 01-load-corpus.py and 02 first"

    db = jammi_ai.connect(f"file://{str(ARTIFACT_DIR)}")
    ensure_source(db, "corpus", str(CORPUS_PARQUET))

    # Encode an audio query (the held-out "sine" query) and search.
    query_wav = (AUDIO_CORPUS_DIR / "queries" / "q_sine.wav").read_bytes()
    query_vec = db.encode_query(model=MODEL, query=query_wav, modality="audio")
    print(f"query embedding dim: {len(query_vec)}")

    results = db.search("corpus", query=query_vec, k=5)  # pyarrow.Table
    assert results.num_rows > 0, "search must return a non-empty top-K"
    print(f"top-{results.num_rows} for q_sine: {results.column('clip_id').to_pylist()}")

    print("03-search: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
