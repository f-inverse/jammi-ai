"""Image-to-image search over a tiny synthetic corpus with an OpenCLIP model.

End-to-end walkthrough: load a small image corpus -> generate vision
embeddings -> run cosine nearest-neighbour search with an image query ->
evaluate retrieval quality (Recall@K / MRR) against a held-out golden set ->
adapt the vision tower itself on image triplets.

The numbered scripts (`01-load-corpus.py` ... `04-eval.py`) decompose the
search-and-eval flow step by step; this file runs every phase in one process
and is the version wired into `tests/cookbook_smoke.py`.

Fine-tuning. Phase 5 trains LoRA adapters INSIDE the OpenCLIP vision tower —
`target_modules=["in_proj", "c_fc"]` names the transformer block's fused-QKV
projection and the MLP's first linear, so the tower's own representation moves,
rather than a projection head learning on top of a frozen tower (the audio
recipe shows that cheaper mode). The triplets are synthetic — positive = a
same-family sibling image, negative = an image from another family — but what
makes an image a "positive" (a redraw, another view of the same object, an
augmentation) is entirely the caller's data; the trainer only minimizes the
contrastive objective over whatever images you pair.

Phase 6 is the refusal: a `target_modules` list that names no site on this
tower fails the JOB rather than quietly training zero parameters, and the
failure message carries this tower's own site names so the fix is a paste.

Model. The default model is the hermetic `tiny_open_clip` fixture so the
recipe runs offline in CI in well under 60s. The federal use case driving
this recipe is PatentCLIP — to run against it, set:

    JAMMI_IMAGE_MODEL=patentclip/PatentCLIP_Vit_B

`patentclip/PatentCLIP_Vit_B` is downloaded from the Hugging Face Hub on
first use and produces 512-dim L2-normalized embeddings. Any OpenCLIP-format
vision model works the same way (OpenAI CLIP, LAION ViT-B/32, ...).

Run with `python cookbook/recipes/image_search/example.py`. Exits 0 on success.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import jammi
from jammi.errors import TrainingError

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURES = REPO_ROOT / "cookbook" / "fixtures"
IMAGE_CORPUS_DIR = FIXTURES / "tiny_image_corpus"
GOLDEN_PATH = FIXTURES / "tiny_image_golden.json"

# Default to the hermetic local fixture so CI runs offline. Override with
# JAMMI_IMAGE_MODEL=patentclip/PatentCLIP_Vit_B (the federal use case) or any
# other OpenCLIP-format model ID / `local:<path>`.
DEFAULT_MODEL = f"local:{FIXTURES / 'tiny_open_clip'}"
MODEL = os.environ.get("JAMMI_IMAGE_MODEL", DEFAULT_MODEL)


def load_corpus_table() -> pa.Table:
    """Read every `img_*.png` in the corpus dir into an Arrow table with the
    inline image bytes the embedding pipeline consumes.

    Schema: `image_id` (utf8 key), `image` (binary, the raw PNG bytes).
    """
    rows = sorted(IMAGE_CORPUS_DIR.glob("img_*.png"))
    assert rows, f"no corpus images under {IMAGE_CORPUS_DIR}"
    ids = [p.stem for p in rows]
    blobs = [p.read_bytes() for p in rows]
    return pa.table(
        {
            "image_id": pa.array(ids, type=pa.utf8()),
            "image": pa.array(blobs, type=pa.binary()),
        }
    )


def build_image_golden(json_path: Path) -> pa.Table:
    """Flatten the per-query golden JSON into the (query_id, query_image,
    relevant_id) shape `db.eval_embeddings` consumes in image mode.

    The presence of a `query_image` (binary) column is what switches the eval
    runner from text-query to image-query encoding.
    """
    queries = json.loads(json_path.read_text())
    query_ids: list[str] = []
    query_images: list[bytes] = []
    relevant_ids: list[str] = []
    for q in queries:
        image_bytes = (IMAGE_CORPUS_DIR / q["query_image"]).read_bytes()
        for rid in q["relevant_ids"]:
            query_ids.append(q["query_id"])
            query_images.append(image_bytes)
            relevant_ids.append(str(rid))
    return pa.table(
        {
            "query_id": pa.array(query_ids, type=pa.utf8()),
            "query_image": pa.array(query_images, type=pa.binary()),
            "relevant_id": pa.array(relevant_ids, type=pa.utf8()),
        }
    )


def corpus_by_family() -> dict[str, list[tuple[str, bytes]]]:
    """Group the corpus images by shape family (the token in
    `img_<family>_<idx>.png`), preserving a deterministic order."""
    families: dict[str, list[tuple[str, bytes]]] = {}
    for path in sorted(IMAGE_CORPUS_DIR.glob("img_*.png")):
        stem = path.stem  # img_circle_0
        family = stem[len("img_") :].rsplit("_", 1)[0]  # -> circle
        families.setdefault(family, []).append((stem, path.read_bytes()))
    assert families, f"no corpus images under {IMAGE_CORPUS_DIR}"
    return families


def build_image_triplets() -> pa.Table:
    """Synthetic `(anchor, positive, negative)` image triplets.

    For each image: positive = the next image in the same shape family,
    negative = an image from a different family. All three columns are raw
    image bytes — the same encoded PNGs the embedding pipeline consumes. The
    trainer encodes them through the vision tower and minimizes the triplet
    loss; the *meaning* of the pairing is this builder's choice, not the
    trainer's. Same column shape as the audio recipe's triplets — only the
    payload's modality differs, and `task=` (not the bytes) is what says which.
    """
    families = corpus_by_family()
    fam_names = list(families)
    anchors: list[bytes] = []
    positives: list[bytes] = []
    negatives: list[bytes] = []
    for fi, fam in enumerate(fam_names):
        images = families[fam]
        neg_images = families[fam_names[(fi + 1) % len(fam_names)]]
        for ci, (_, anchor) in enumerate(images):
            anchors.append(anchor)
            positives.append(images[(ci + 1) % len(images)][1])
            negatives.append(neg_images[ci % len(neg_images)][1])
    return pa.table(
        {
            "anchor": pa.array(anchors, type=pa.binary()),
            "positive": pa.array(positives, type=pa.binary()),
            "negative": pa.array(negatives, type=pa.binary()),
        }
    )


def main() -> int:
    print(f"image_search: model = {MODEL}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db = jammi.connect(f"file://{str(tmp_path)}")

        # 1. Load the corpus images into a Parquet source (inline image bytes).
        corpus_parquet = tmp_path / "corpus.parquet"
        pq.write_table(load_corpus_table(), corpus_parquet)
        db.add_source("corpus", url=str(corpus_parquet), format="parquet")

        # 2. Generate vision embeddings over the `image` column. The model is
        #    auto-detected from its OpenCLIP config; output is L2-normalized.
        db.generate_embeddings(
            source="corpus",
            model=MODEL,
            columns=["image"],
            key="image_id",
            modality="image",
        )

        # 3. Encode a single image query and run cosine ANN search.
        query_png = (IMAGE_CORPUS_DIR / "queries" / "q_circle.png").read_bytes()
        query_vec = db.encode_query(model=MODEL, query=query_png, modality="image")
        assert query_vec, "query embedding must be non-empty"
        print(f"query embedding dim: {len(query_vec)}")

        results = db.search("corpus", query=query_vec, k=5)  # pyarrow.Table
        assert results.num_rows > 0, "search must return a non-empty top-K"
        top_ids = results.column("image_id").to_pylist()
        print(f"top-{results.num_rows} for q_circle: {top_ids}")

        # 4. Evaluate retrieval quality against the held-out golden set. The
        #    eval encodes each golden `query_image`, searches, and reports
        #    Recall@K / MRR per query and in aggregate. We measure and report
        #    — we do NOT assert a quality target (the fixture model has random
        #    weights; real numbers come from a real model like PatentCLIP).
        golden_parquet = tmp_path / "golden.parquet"
        pq.write_table(build_image_golden(GOLDEN_PATH), golden_parquet)
        db.add_source("golden", url=str(golden_parquet), format="parquet")

        metrics = db.eval_embeddings(
            source="corpus",
            golden_source="golden.public.golden",
            k=5,
        )

        aggregate = metrics["aggregate"]
        print("aggregate retrieval metrics:")
        for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
            value = aggregate[key]
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"
            print(f"  {key:<16} {value:.4f}")

        per_query = metrics["per_query"]
        assert len(per_query) > 0, "per_query must carry one record per query"
        print(f"per_query: {len(per_query)} records")

        # 5. Adapt the vision TOWER on image triplets. A non-empty
        #    `target_modules` puts LoRA inside the OpenCLIP vision
        #    transformer's own sites — `in_proj` is the fused-QKV projection,
        #    `c_fc` the MLP's first linear — so the tower's representation
        #    itself moves. (The audio recipe shows both modes side by side:
        #    an empty list instead trains a projection head on a FROZEN tower.)
        #    The site names are this architecture's; a name from another
        #    architecture's recipe is refused in step 6.
        triplets_parquet = tmp_path / "image_triplets.parquet"
        pq.write_table(build_image_triplets(), triplets_parquet)
        db.add_source("triplets", url=str(triplets_parquet), format="parquet")

        job = db.fine_tune(
            source="triplets",
            base_model=MODEL,
            columns=["anchor", "positive", "negative"],
            method="lora",
            task="image_embedding",
            target_modules=["in_proj", "c_fc"],
            lora_rank=4,
            learning_rate=5e-3,
            epochs=2,
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
        print(f"fine-tuned image model: {tuned_model}")

        # The adapted model registers in the catalog under the MEDIA task, so
        # `search`'s model resolution finds an image encoder, not a text one.
        # This is the most the *client* surface says about the artifact: the
        # saved adapter's kind (encoder adapters, and which tower they were
        # injected into) is engine-internal and is pinned by the engine's own
        # integration tests, not readable from `describe_model` here. What this
        # recipe can prove at the consumer surface is the next check: the
        # served embedding actually moved.
        described = db.describe_model(tuned_model)
        assert described is not None, f"{tuned_model} missing from the catalog"
        assert described["task"] == "image_embedding", (
            f"fine-tuned model registered under the wrong task: {described}"
        )

        # The adapted tower serves under the new model id: re-encode the SAME
        # query image through it and compare against the base encoding
        # (`query_vec`, from step 3). Vector change is the invariant the
        # adapter guarantees — a LoRA delta that was trained but silently
        # dropped at serve time would leave these two vectors identical. We
        # assert on the vectors, not on the coarse top-k metrics: on this tiny
        # eval set the rankings rarely flip even when the vectors move. (With
        # the random-weight fixture the *direction* of the change is not
        # meaningful; a real OpenCLIP checkpoint is where tuning lifts quality.
        # We assert change, not improvement.)
        tuned_query_vec = db.encode_query(
            model=tuned_model, query=query_png, modality="image"
        )
        assert len(tuned_query_vec) == len(query_vec), (
            "tuned query embedding dim must match the base dim "
            f"(base={len(query_vec)}, tuned={len(tuned_query_vec)})"
        )
        max_abs_diff = max(abs(b - t) for b, t in zip(query_vec, tuned_query_vec))
        print(f"query embedding max |Δ| (base vs tuned): {max_abs_diff:.6f}")
        assert max_abs_diff > 1e-4, (
            "the tower adapter should change the served image embedding: the "
            "tuned query vector is identical to the base vector "
            f"(max |Δ| = {max_abs_diff:.2e} <= 1e-4) — the adapter was trained "
            "but is not being applied when the model is served"
        )

        # 6. The refusal. `q_proj` is a real site name on plenty of decoder
        #    checkpoints and on nothing in an OpenCLIP tower — exactly the
        #    plausible-but-wrong string carried over from another
        #    architecture's recipe. Selecting no site would train zero
        #    parameters and publish an adapter that changes nothing, so the
        #    engine fails the JOB instead, and the message names this tower's
        #    real sites so the fix is a paste, not a search.
        refused = db.fine_tune(
            source="triplets",
            base_model=MODEL,
            columns=["anchor", "positive", "negative"],
            method="lora",
            task="image_embedding",
            target_modules=["q_proj"],
            lora_rank=4,
            epochs=1,
            batch_size=4,
            warmup_steps=0,
            validation_fraction=0.0,
            early_stopping_metric="train_loss",
        )
        try:
            refused.wait()
        except TrainingError as exc:
            message = str(exc)
            print(f"refused, as designed: {message}")
            assert "q_proj" in message, (
                f"the refusal must echo the submitted selector: {message}"
            )
            assert "in_proj" in message, (
                "the refusal must name this tower's real site names so the "
                f"caller can paste one: {message}"
            )
        else:
            raise AssertionError(
                "a target_modules list matching no site on the vision tower "
                "must FAIL the job, never publish an empty adapter under a "
                "fine-tuned model id"
            )

    print("image_search: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
