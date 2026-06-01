"""Step 2 — generate audio embeddings over the corpus.

Connects to the shared persistent artifact dir, registers the `corpus` source
written by step 01, and builds an L2-normalized audio-embedding index over the
`audio` column. The backend owns decode -> resample -> log-mel -> forward. The
embedding table + ANN sidecar persist in the artifact dir, so step 03 (search)
and step 04 (eval) reopen the same dir and reuse them.

The model is auto-detected from its CLAP config. Default is the hermetic
`tiny_clap` fixture; set JAMMI_AUDIO_MODEL=<hf-repo-id> for a real CLAP model.
"""

from __future__ import annotations

import jammi_ai

from _shared import ARTIFACT_DIR, CORPUS_PARQUET, MODEL, ensure_source


def main() -> int:
    assert CORPUS_PARQUET.exists(), "run 01-load-corpus.py first"

    db = jammi_ai.connect(
        artifact_dir=str(ARTIFACT_DIR),
        gpu_device=-1,
        inference_batch_size=8,
    )
    ensure_source(db, "corpus", str(CORPUS_PARQUET))

    print(f"generating audio embeddings with {MODEL} ...")
    db.generate_audio_embeddings(
        source="corpus",
        model=MODEL,
        audio_column="audio",
        key="clip_id",
    )

    print("02-generate-embeddings: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
