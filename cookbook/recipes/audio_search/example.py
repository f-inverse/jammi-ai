"""Audio-to-audio search over a tiny synthetic corpus with a CLAP model.

End-to-end walkthrough: load a small audio corpus -> generate audio
embeddings -> run cosine nearest-neighbour search with an audio query ->
evaluate retrieval quality (Recall@K / MRR) against a held-out golden set ->
domain-tune a projection head on audio triplets and re-evaluate.

The numbered scripts (`01-load-corpus.py` ... `04-eval.py`) decompose the
search-and-eval flow step by step; this file runs every phase in one process
and is the version wired into `tests/cookbook_smoke.py`.

Fine-tuning. Phase 5 trains a lightweight projection head on a *frozen* CLAP
audio tower from `(anchor, positive, negative)` audio triplets and re-runs the
eval, showing the tuned embeddings differ from the base. The triplets here are
synthetic — positive = a same-family clip, negative = a different-family clip —
but what makes a clip a "positive" (augmentation-similar, or co-occurring-
complementary) is entirely the caller's data; the trainer only minimizes the
contrastive objective over whatever clips you pair.

Model. The default model is the hermetic `htsat_clap_tiny` fixture so the recipe
runs offline in CI in well under 60s. Any HuggingFace CLAP audio model works the
same way — point `JAMMI_AUDIO_MODEL` at a Hugging Face repo id or
`local:<path>` whose `config.json` declares `model_type = "clap_audio_model"`
(or lists `ClapModel` / `ClapAudioModelWithProjection` in `architectures`) and
whose checkpoint exposes the `audio_model.audio_encoder.*` + `audio_projection.*`
tower keys, alongside a `preprocessor_config.json` feature-extractor config:

    JAMMI_AUDIO_MODEL=laion/clap-htsat-fused   # (illustrative)

The fixture has random weights, so its embeddings are garbage and the eval
numbers are not meaningful — the recipe measures and reports them to exercise
the full path; real retrieval quality comes from a real CLAP checkpoint.

Run with `python cookbook/recipes/audio_search/example.py`. Exits 0 on success.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import jammi_ai

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURES = REPO_ROOT / "cookbook" / "fixtures"
AUDIO_CORPUS_DIR = FIXTURES / "tiny_audio_corpus"
GOLDEN_PATH = FIXTURES / "tiny_audio_golden.json"

# Default to the hermetic local fixture so CI runs offline. Override with
# JAMMI_AUDIO_MODEL=<hf-repo-id> or `local:<path>` for any CLAP-format model.
DEFAULT_MODEL = f"local:{FIXTURES / 'htsat_clap_tiny'}"
MODEL = os.environ.get("JAMMI_AUDIO_MODEL", DEFAULT_MODEL)


def load_corpus_table() -> pa.Table:
    """Read every `clip_*.wav` in the corpus dir into an Arrow table with the
    inline audio bytes the embedding pipeline consumes.

    Schema: `clip_id` (utf8 key), `audio` (binary, the raw WAV bytes).
    """
    rows = sorted(AUDIO_CORPUS_DIR.glob("clip_*.wav"))
    assert rows, f"no corpus clips under {AUDIO_CORPUS_DIR}"
    ids = [p.stem for p in rows]
    blobs = [p.read_bytes() for p in rows]
    return pa.table(
        {
            "clip_id": pa.array(ids, type=pa.utf8()),
            "audio": pa.array(blobs, type=pa.binary()),
        }
    )


def build_audio_golden(json_path: Path) -> pa.Table:
    """Flatten the per-query golden JSON into the (query_id, query_audio,
    relevant_id) shape `db.eval_embeddings` consumes in audio mode.

    The presence of a `query_audio` (binary) column is what switches the eval
    runner from text/image-query to audio-query encoding.
    """
    queries = json.loads(json_path.read_text())
    query_ids: list[str] = []
    query_audios: list[bytes] = []
    relevant_ids: list[str] = []
    for q in queries:
        audio_bytes = (AUDIO_CORPUS_DIR / q["query_audio"]).read_bytes()
        for rid in q["relevant_ids"]:
            query_ids.append(q["query_id"])
            query_audios.append(audio_bytes)
            relevant_ids.append(str(rid))
    return pa.table(
        {
            "query_id": pa.array(query_ids, type=pa.utf8()),
            "query_audio": pa.array(query_audios, type=pa.binary()),
            "relevant_id": pa.array(relevant_ids, type=pa.utf8()),
        }
    )


def corpus_by_family() -> dict[str, list[tuple[str, bytes]]]:
    """Group the corpus clips by timbre family (the token in
    `clip_<family>_<idx>.wav`), preserving a deterministic order."""
    families: dict[str, list[tuple[str, bytes]]] = {}
    for path in sorted(AUDIO_CORPUS_DIR.glob("clip_*.wav")):
        stem = path.stem  # clip_sine_0
        family = stem[len("clip_") :].rsplit("_", 1)[0]  # -> sine
        families.setdefault(family, []).append((stem, path.read_bytes()))
    assert families, f"no corpus clips under {AUDIO_CORPUS_DIR}"
    return families


def build_audio_triplets() -> pa.Table:
    """Synthetic `(anchor, positive, negative)` audio triplets.

    For each clip: positive = the next clip in the same family, negative = a
    clip from a different family. All three columns are raw audio bytes — the
    same encoded clips the embedding pipeline consumes. The trainer encodes
    them through the frozen audio tower + projection head and minimizes the
    triplet loss; the *meaning* of the pairing is this builder's choice, not
    the trainer's.
    """
    families = corpus_by_family()
    fam_names = list(families)
    anchors: list[bytes] = []
    positives: list[bytes] = []
    negatives: list[bytes] = []
    for fi, fam in enumerate(fam_names):
        clips = families[fam]
        neg_clips = families[fam_names[(fi + 1) % len(fam_names)]]
        for ci, (_, anchor) in enumerate(clips):
            positives.append(clips[(ci + 1) % len(clips)][1])
            negatives.append(neg_clips[ci % len(neg_clips)][1])
            anchors.append(anchor)
    return pa.table(
        {
            "anchor": pa.array(anchors, type=pa.binary()),
            "positive": pa.array(positives, type=pa.binary()),
            "negative": pa.array(negatives, type=pa.binary()),
        }
    )


def aggregate_line(metrics: dict) -> str:
    """One-line summary of the four aggregate retrieval metrics."""
    agg = metrics["aggregate"]
    return "  ".join(
        f"{key}={agg[key]:.4f}"
        for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg")
    )


def main() -> int:
    print(f"audio_search: model = {MODEL}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db = jammi_ai.connect(f"file://{str(tmp_path)}")

        # 1. Load the corpus clips into a Parquet source (inline audio bytes).
        corpus_parquet = tmp_path / "corpus.parquet"
        pq.write_table(load_corpus_table(), corpus_parquet)
        db.add_source("corpus", url=str(corpus_parquet), format="parquet")

        # 2. Generate audio embeddings over the `audio` column. The model is
        #    auto-detected from its CLAP config; the backend owns
        #    decode -> resample -> log-mel -> forward; output is L2-normalized.
        db.generate_embeddings(
            source="corpus",
            model=MODEL,
            columns=["audio"],
            key="clip_id",
            modality="audio",
        )

        # 3. Encode a single audio query and run cosine ANN search.
        query_wav = (AUDIO_CORPUS_DIR / "queries" / "q_sine.wav").read_bytes()
        query_vec = db.encode_query(model=MODEL, query=query_wav, modality="audio")
        assert query_vec, "query embedding must be non-empty"
        print(f"query embedding dim: {len(query_vec)}")

        results = db.search("corpus", query=query_vec, k=5)  # pyarrow.Table
        assert results.num_rows > 0, "search must return a non-empty top-K"
        top_ids = results.column("clip_id").to_pylist()
        print(f"top-{results.num_rows} for q_sine: {top_ids}")

        # 4. Evaluate retrieval quality against the held-out golden set. The
        #    eval encodes each golden `query_audio`, searches, and reports
        #    Recall@K / MRR per query and in aggregate. We measure and report
        #    — we do NOT assert a quality target (the fixture model has random
        #    weights; real numbers come from a real CLAP checkpoint).
        golden_parquet = tmp_path / "golden.parquet"
        pq.write_table(build_audio_golden(GOLDEN_PATH), golden_parquet)
        db.add_source("golden", url=str(golden_parquet), format="parquet")

        base_metrics = db.eval_embeddings(
            source="corpus",
            golden_source="golden.public.golden",
            k=5,
        )

        aggregate = base_metrics["aggregate"]
        print("base aggregate retrieval metrics:")
        for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
            value = aggregate[key]
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"
            print(f"  {key:<16} {value:.4f}")

        per_query = base_metrics["per_query"]
        assert len(per_query) > 0, "per_query must carry one record per query"
        print(f"per_query: {len(per_query)} records")

        # 5. Domain-tune a projection head on audio triplets, then re-evaluate.
        #    Empty target_modules => a trainable projection head on the FROZEN
        #    CLAP audio tower (the cheap, low-risk lightweight mode). The
        #    triplet loss only needs (anchor, positive, negative) clips; the
        #    pairing semantics are ours, above, not the trainer's.
        triplets_parquet = tmp_path / "audio_triplets.parquet"
        pq.write_table(build_audio_triplets(), triplets_parquet)
        db.add_source("triplets", url=str(triplets_parquet), format="parquet")

        job = db.fine_tune(
            source="triplets",
            base_model=MODEL,
            columns=["anchor", "positive", "negative"],
            method="lora",
            task="audio_embedding",
            lora_rank=4,
            learning_rate=1e-3,
            epochs=8,
            batch_size=4,
            warmup_steps=0,
            validation_fraction=0.0,
            early_stopping_metric="train_loss",
        )
        job.wait()
        tuned_model = job.model_id
        assert tuned_model.startswith("jammi:fine-tuned:"), (
            f"unexpected fine-tuned model_id: {tuned_model}"
        )
        print(f"fine-tuned audio model: {tuned_model}")

        # Re-embed the corpus with the tuned model and eval against the same
        # held-out golden set.
        db.generate_embeddings(
            source="corpus",
            model=tuned_model,
            columns=["audio"],
            key="clip_id",
            modality="audio",
        )
        tuned_metrics = db.eval_embeddings(
            source="corpus",
            golden_source="golden.public.golden",
            k=5,
        )
        for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
            value = tuned_metrics["aggregate"][key]
            assert 0.0 <= value <= 1.0, f"tuned {key} out of range: {value}"

        print(f"base : {aggregate_line(base_metrics)}")
        print(f"tuned: {aggregate_line(tuned_metrics)}")

        # The projection head changed the audio embeddings — re-encode the SAME
        # query through the tuned tower+head and compare the vectors to the base
        # encoding (`query_vec`, from step 3). Vector change is the real
        # invariant fine-tuning guarantees: the training loss moves, so the
        # projected embeddings must differ. We assert on the vectors, not on the
        # coarse top-k metrics above — on this tiny eval set the rankings rarely
        # flip even when the vectors move, so a metric-inequality check passes
        # only intermittently. (With the random-weight fixture the *direction*
        # of the change is not meaningful; a real CLAP checkpoint is where
        # tuning lifts quality. We assert change, not improvement.)
        tuned_query_vec = db.encode_query(model=tuned_model, query=query_wav, modality="audio")
        assert len(tuned_query_vec) == len(query_vec), (
            "tuned query embedding dim must match the base dim "
            f"(base={len(query_vec)}, tuned={len(tuned_query_vec)})"
        )
        max_abs_diff = max(
            abs(b - t) for b, t in zip(query_vec, tuned_query_vec)
        )
        print(f"query embedding max |Δ| (base vs tuned): {max_abs_diff:.6f}")
        assert max_abs_diff > 1e-4, (
            "fine-tuned audio embeddings should differ from base: the tuned "
            "query vector is identical to the base vector "
            f"(max |Δ| = {max_abs_diff:.2e} <= 1e-4) — the projection head did "
            "not change the embedding"
        )

    print("audio_search: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
