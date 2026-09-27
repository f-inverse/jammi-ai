"""Search an image corpus with a text query — the CLIP text tower against the
vision tower's index.

An OpenCLIP-format checkpoint carries a vision tower and a text tower that
project into one shared latent space, so a text query encoded by the text tower
searches an image index directly: no separate text encoder, no projection
bridge. This recipe runs the three steps the guide's "Search Text Against
Images (Cross-Modal)" page teaches:

1. index the images with the vision tower (`generate_embeddings(...,
   modality="image")`);
2. encode a text query with the SAME model's text tower (`encode_query(...)`,
   text is the default modality);
3. `search()` the image index with that vector.

What it checks. The ranking a cross-modal `search` returns is exactly the
cosine ranking in the shared space: every corpus image is encoded on its own
through the vision tower (`encode_query(..., modality="image")`), scored
against the text vector in plain Python, and the top-k — ids and scores —
must equal the search's. That holds for any checkpoint.

Model. The default is the hermetic `tiny_open_clip` fixture, so the recipe runs
offline in CI in seconds. Its weights are random: the text and image vectors
share a space, but "a circle" means nothing to it, so with the fixture the
recipe asserts the mechanics above and prints the ranking without judging it.
Point `JAMMI_CROSS_MODAL_MODEL` at a trained OpenCLIP checkpoint to see the
semantics, e.g.:

    JAMMI_CROSS_MODAL_MODEL=laion/CLIP-ViT-B-32-laion2B-s34B-b79K

(downloaded from the Hugging Face Hub on first use). With a trained model the
recipe also prints, for each shape family's prompt, the family of the
top-ranked image.

Run with `python cookbook/recipes/cross_modal_search/example.py`. Exits 0 on
success.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import jammi
from jammi_cookbook import fixtures

IMAGE_CORPUS_DIR = fixtures.path("tiny_image_corpus")
DEFAULT_MODEL = fixtures.model("tiny_open_clip")
MODEL = os.environ.get("JAMMI_CROSS_MODAL_MODEL", DEFAULT_MODEL)
K = 5
FAMILIES = ["circle", "triangle", "square", "hexagon", "grating"]


def corpus() -> list[tuple[str, bytes]]:
    """Every `img_<family>_<n>.png` of the synthetic corpus, as (id, PNG bytes)."""
    rows = [(p.stem, p.read_bytes()) for p in sorted(IMAGE_CORPUS_DIR.glob("img_*.png"))]
    assert rows, f"no corpus images under {IMAGE_CORPUS_DIR}"
    return rows


def family(image_id: str) -> str:
    """The shape family an image id names: `img_circle_0` -> `circle`."""
    return image_id[len("img_") :].rsplit("_", 1)[0]


def cosine_top_k(
    query: list[float], vectors: dict[str, list[float]], k: int
) -> list[tuple[str, float]]:
    """The `k` ids nearest `query` by cosine similarity, best first."""

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

    scored = sorted(
        ((cosine(query, v), i) for i, v in vectors.items()), key=lambda s: (-s[0], s[1])
    )
    return [(i, s) for s, i in scored[:k]]


def main() -> int:
    print(f"cross_modal_search: model = {MODEL}")
    images = corpus()
    with tempfile.TemporaryDirectory() as tmp, jammi.connect(f"file://{tmp}") as db:
        # 1. Index the image corpus with the vision tower.
        corpus_parquet = Path(tmp) / "figures.parquet"
        pq.write_table(
            pa.table(
                {
                    "figure_id": pa.array([i for i, _ in images], type=pa.utf8()),
                    "image": pa.array([b for _, b in images], type=pa.binary()),
                }
            ),
            corpus_parquet,
        )
        db.add_source("figures", url=str(corpus_parquet), format="parquet")
        db.generate_embeddings(
            source="figures",
            model=MODEL,
            columns=["image"],
            key="figure_id",
            modality="image",
        )

        # 2. Embed a text query with the same model's text tower.
        prompt = "a drawing of a circle"
        query_vec = db.encode_query(model=MODEL, query=prompt)

        # 3. Search the image embeddings with the text vector.
        results = db.search("figures", query=query_vec, k=K)
        ranked = list(
            zip(results.column("figure_id").to_pylist(), results.column("similarity").to_pylist())
        )
        print(f"text query {prompt!r} -> top-{K} images:")
        for image_id, similarity in ranked:
            print(f"  {image_id:<18} similarity {similarity:.4f}")

        # The ranking is the cosine ranking in the shared space: each image
        # encoded on its own by the vision tower, scored against the text
        # vector here. Same ids, same order, same scores.
        image_vectors = {
            image_id: db.encode_query(model=MODEL, query=png, modality="image")
            for image_id, png in images
        }
        assert all(len(v) == len(query_vec) for v in image_vectors.values()), (
            "the text and vision towers project into one space of one width"
        )
        oracle = cosine_top_k(query_vec, image_vectors, K)
        assert [i for i, _ in ranked] == [i for i, _ in oracle], (ranked, oracle)
        for (_, got), (_, want) in zip(ranked, oracle):
            assert abs(got - want) < 1e-4, (ranked, oracle)
        print(f"top-{K} equals the cosine ranking in the shared space (dim {len(query_vec)})")

        if MODEL != DEFAULT_MODEL:
            # A trained checkpoint: which family each family's prompt ranks first.
            for name in FAMILIES:
                vec = db.encode_query(model=MODEL, query=f"a drawing of a {name}")
                best = db.search("figures", query=vec, k=1).column("figure_id").to_pylist()[0]
                print(f"  'a drawing of a {name}' -> {best} ({family(best)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
