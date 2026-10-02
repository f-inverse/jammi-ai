"""Search an image corpus with a text query: a CLIP text tower against its vision tower's index.

Run with `python cookbook/recipes/cross_modal_search/example.py`, or a step at
a time as a notebook: each `# %%` cell is one step.
"""

# %% [markdown]
# An OpenCLIP-format checkpoint carries a vision tower and a text tower that
# project into one shared space, so a text query encoded by the text tower
# searches an image index directly: no separate text encoder, no projection
# bridge. `JAMMI_CROSS_MODAL_MODEL` names the checkpoint; the default is the
# random-weight `tiny_open_clip` fixture.

# %%
import math
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import jammi
from jammi_cookbook import fixtures

DEFAULT_MODEL = fixtures.model("tiny_open_clip")
MODEL = os.environ.get("JAMMI_CROSS_MODAL_MODEL", DEFAULT_MODEL)
K = 5
FAMILIES = ["circle", "triangle", "square", "hexagon", "grating"]
print(f"model: {MODEL}")

home = Path(tempfile.mkdtemp())
db = jammi.connect(f"file://{home}")

# %% [markdown]
# ## Index the images with the vision tower
#
# The corpus is 20 synthetic drawings in five shape families, held as PNG
# bytes in a Parquet source. `generate_embeddings` with `modality="image"`
# runs the vision tower over them.

# %%
images = [
    (p.stem, p.read_bytes()) for p in sorted(fixtures.path("tiny_image_corpus").glob("img_*.png"))
]
assert images, "no corpus images"

pq.write_table(
    pa.table(
        {
            "figure_id": pa.array([i for i, _ in images], type=pa.utf8()),
            "image": pa.array([b for _, b in images], type=pa.binary()),
        }
    ),
    home / "figures.parquet",
)
db.add_source("figures", url=str(home / "figures.parquet"), format="parquet")
db.generate_embeddings(
    source="figures", model=MODEL, columns=["image"], key="figure_id", modality="image"
)

# %% [markdown]
# ## Search them with words
#
# `encode_query` runs the same model's text tower (text is the default
# modality), and `search` ranks the image index against that vector.

# %%
prompt = "a drawing of a circle"
query_vec = db.encode_query(model=MODEL, query=prompt)

results = db.search("figures", query=query_vec, k=K)
ranked = list(
    zip(results.column("figure_id").to_pylist(), results.column("similarity").to_pylist())
)
print(f"text query {prompt!r} -> top-{K} images:")
for image_id, similarity in ranked:
    print(f"  {image_id:<18} similarity {similarity:.4f}")

# %% [markdown]
# ## The ranking is the shared space's
#
# Encode every image on its own through the vision tower, score each against
# the text vector by cosine in plain Python, and the top five — ids and scores
# — are the search's. That holds for any checkpoint.

# %%
def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


image_vectors = {
    image_id: db.encode_query(model=MODEL, query=png, modality="image") for image_id, png in images
}
assert all(len(v) == len(query_vec) for v in image_vectors.values()), (
    "the text and vision towers project into one space of one width"
)
scored = sorted(
    ((cosine(query_vec, v), i) for i, v in image_vectors.items()), key=lambda s: (-s[0], s[1])
)
oracle = [(i, s) for s, i in scored[:K]]

assert [i for i, _ in ranked] == [i for i, _ in oracle], (ranked, oracle)
for (_, got), (_, want) in zip(ranked, oracle):
    assert abs(got - want) < 1e-4, (ranked, oracle)
print(f"top-{K} equals the cosine ranking in the shared space (dim {len(query_vec)})")

# %% [markdown]
# ## With a trained checkpoint
#
# A trained model ranks first an image of the family its prompt names. With the
# random-weight fixture this step has nothing to show and prints nothing.

# %%
if MODEL != DEFAULT_MODEL:
    for name in FAMILIES:
        vec = db.encode_query(model=MODEL, query=f"a drawing of a {name}")
        best = db.search("figures", query=vec, k=1).column("figure_id").to_pylist()[0]
        print(f"'a drawing of a {name}' -> {best}")

# %%
db.close()
