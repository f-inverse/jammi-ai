"""Step 1 — load the synthetic audio corpus into a Jammi source.

Reads the 20 committed mono WAV clips under
`cookbook/fixtures/tiny_audio_corpus/`, writes them (with inline audio bytes)
to a Parquet file in the shared workdir, and registers it as the `corpus`
source. Steps 02-04 reuse the same workdir.

Run the steps in order:

    python cookbook/recipes/audio_search/01-load-corpus.py
    python cookbook/recipes/audio_search/02-generate-embeddings.py
    python cookbook/recipes/audio_search/03-search.py
    python cookbook/recipes/audio_search/04-eval.py
"""

from __future__ import annotations

import pyarrow.parquet as pq

from _shared import ARTIFACT_DIR, CORPUS_PARQUET, WORKDIR, load_corpus_table


def main() -> int:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    table = load_corpus_table()
    pq.write_table(table, CORPUS_PARQUET)

    print(f"loaded {table.num_rows} clips -> {CORPUS_PARQUET}")
    print("columns:", table.column_names)
    print("01-load-corpus: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
