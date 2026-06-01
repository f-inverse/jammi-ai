"""Step 3 — search the index with an audio query.

Encodes a held-out query clip into a vector and runs cosine nearest-neighbour
search over the corpus index built in step 02. Prints the top-K clip IDs.

The `SearchBuilder` returned by `db.search(...)` supports `.filter()`,
`.select()`, `.sort()`, `.limit()`, `.join()` and `.annotate()` before `.run()`
— here we keep it to a plain top-K.
"""

from __future__ import annotations

import jammi_ai

from _shared import ARTIFACT_DIR, AUDIO_CORPUS_DIR, CORPUS_PARQUET, MODEL, ensure_source


def main() -> int:
    assert CORPUS_PARQUET.exists(), "run 01-load-corpus.py and 02 first"

    db = jammi_ai.connect(
        artifact_dir=str(ARTIFACT_DIR),
        gpu_device=-1,
        inference_batch_size=8,
    )
    ensure_source(db, "corpus", str(CORPUS_PARQUET))

    # Encode an audio query (the held-out "sine" query) and search.
    query_wav = (AUDIO_CORPUS_DIR / "queries" / "q_sine.wav").read_bytes()
    query_vec = db.encode_audio_query(MODEL, query_wav)
    print(f"query embedding dim: {len(query_vec)}")

    results = db.search("corpus", query=query_vec, k=5).run()
    assert results.num_rows > 0, "search must return a non-empty top-K"
    print(f"top-{results.num_rows} for q_sine: {results.column('clip_id').to_pylist()}")

    print("03-search: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
