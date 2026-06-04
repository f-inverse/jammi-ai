"""Step 2 — generate audio embeddings over the corpus.

Connects to the shared persistent artifact dir, registers the `corpus` source
written by step 01, and builds an L2-normalized audio-embedding index over the
`audio` column. The backend owns decode -> resample -> CLAP fusion log-mel ->
HTSAT-Swin tower forward. The
embedding table + ANN sidecar persist in the artifact dir, so step 03 (search)
and step 04 (eval) reopen the same dir and reuse them.

The model is auto-detected from its CLAP config. Default is the hermetic
`htsat_clap_tiny` fixture; set JAMMI_AUDIO_MODEL=<hf-repo-id> for a real CLAP model.
"""

from __future__ import annotations

import jammi_ai

from _shared import ARTIFACT_DIR, CORPUS_PARQUET, MODEL, ensure_source


def main() -> int:
    assert CORPUS_PARQUET.exists(), "run 01-load-corpus.py first"

    db = jammi_ai.connect(f"file://{str(ARTIFACT_DIR)}")
    ensure_source(db, "corpus", str(CORPUS_PARQUET))

    print(f"generating audio embeddings with {MODEL} ...")
    db.generate_embeddings(
            source="corpus",
            model=MODEL,
            columns=["audio"],
            key="clip_id",
            modality="audio",
        )

    print("02-generate-embeddings: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
