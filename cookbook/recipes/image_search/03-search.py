"""Step 3 — search the index with an image query.

Encodes a held-out query image into a vector and runs cosine nearest-neighbour
search over the corpus index built in step 02. Prints the top-K image IDs.

The `SearchBuilder` returned by `db.search(...)` supports `.filter()`,
`.select()`, `.sort()`, `.limit()`, `.join()` and `.annotate()` before `.run()`
— here we keep it to a plain top-K.
"""

from __future__ import annotations

import jammi_ai

from _shared import ARTIFACT_DIR, CORPUS_PARQUET, IMAGE_CORPUS_DIR, MODEL, ensure_source


def main() -> int:
    assert CORPUS_PARQUET.exists(), "run 01-load-corpus.py and 02 first"

    db = jammi_ai.connect(
        artifact_dir=str(ARTIFACT_DIR),
        gpu_device=-1,
        inference_batch_size=8,
    )
    ensure_source(db, "corpus", str(CORPUS_PARQUET))

    # Encode an image query (the held-out "circle" query) and search.
    query_png = (IMAGE_CORPUS_DIR / "queries" / "q_circle.png").read_bytes()
    query_vec = db.encode_image_query(MODEL, query_png)
    print(f"query embedding dim: {len(query_vec)}")

    results = db.search("corpus", query=query_vec, k=5).run()
    assert results.num_rows > 0, "search must return a non-empty top-K"
    print(f"top-{results.num_rows} for q_circle: {results.column('image_id').to_pylist()}")

    print("03-search: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
