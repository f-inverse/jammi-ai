"""Image-to-image search with an OpenCLIP model: index, search, evaluate, adapt the tower.

Run with `python cookbook/recipes/image_search/example.py`, or a step at a time
as a notebook: each `# %%` cell is one step.
"""

# %% [markdown]
# `JAMMI_IMAGE_MODEL` names the checkpoint, any OpenCLIP-format model id or
# `local:<path>`; the default is the random-weight `tiny_open_clip` fixture,
# which runs offline in seconds.

# %%
import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import jammi
from jammi.errors import TrainingError
from jammi_cookbook import fixtures

IMAGE_CORPUS_DIR = fixtures.path("tiny_image_corpus")
DEFAULT_MODEL = fixtures.model("tiny_open_clip")
MODEL = os.environ.get("JAMMI_IMAGE_MODEL", DEFAULT_MODEL)
print(f"model: {MODEL}")

home = Path(tempfile.mkdtemp())
db = jammi.connect(f"file://{home}")

# %% [markdown]
# ## Load the images
#
# The corpus is 20 synthetic 224×224 PNGs in five shape families, held inline
# as bytes in a Parquet source: `image_id`, and `image` with the raw PNG.

# %%
paths = sorted(IMAGE_CORPUS_DIR.glob("img_*.png"))
assert paths, f"no corpus images under {IMAGE_CORPUS_DIR}"
pq.write_table(
    pa.table(
        {
            "image_id": pa.array([p.stem for p in paths], type=pa.utf8()),
            "image": pa.array([p.read_bytes() for p in paths], type=pa.binary()),
        }
    ),
    home / "corpus.parquet",
)
db.add_source("corpus", url=str(home / "corpus.parquet"), format="parquet")

# %% [markdown]
# ## Embed them
#
# `generate_embeddings` with `modality="image"` runs the vision tower over the
# `image` column. The encoder is read from the checkpoint's OpenCLIP config,
# preprocesses each image as the model's `preprocess_cfg` says, and writes
# L2-normalized vectors.

# %%
db.generate_embeddings(
    source="corpus", model=MODEL, columns=["image"], key="image_id", modality="image"
)

# %% [markdown]
# ## Search with an image
#
# A query image is encoded by the same tower, and `search` returns its
# nearest corpus images by cosine similarity.

# %%
query_png = (IMAGE_CORPUS_DIR / "queries" / "q_circle.png").read_bytes()
query_vec = db.encode_query(model=MODEL, query=query_png, modality="image")
assert query_vec, "query embedding must be non-empty"
print(f"query embedding dim: {len(query_vec)}")

results = db.search("corpus", query=query_vec, k=5)
assert results.num_rows > 0, "search must return a non-empty top-K"
print(f"top-{results.num_rows} for q_circle: {results.column('image_id').to_pylist()}")

# %% [markdown]
# ## Measure retrieval quality
#
# The golden set holds a held-out query image per family and the corpus images
# of that family. A `query_image` (binary) column in place of `query_text` is
# what switches `eval_embeddings` to image queries. The numbers are reported,
# not judged: the fixture's weights are random, and a real checkpoint is where
# they mean something.

# %%
query_ids, query_images, relevant_ids = [], [], []
for q in json.loads(fixtures.path("tiny_image_golden.json").read_text()):
    image_bytes = (IMAGE_CORPUS_DIR / q["query_image"]).read_bytes()
    for rid in q["relevant_ids"]:
        query_ids.append(q["query_id"])
        query_images.append(image_bytes)
        relevant_ids.append(str(rid))
pq.write_table(
    pa.table(
        {
            "query_id": pa.array(query_ids, type=pa.utf8()),
            "query_image": pa.array(query_images, type=pa.binary()),
            "relevant_id": pa.array(relevant_ids, type=pa.utf8()),
        }
    ),
    home / "golden.parquet",
)
db.add_source("golden", url=str(home / "golden.parquet"), format="parquet")

metrics = db.eval_embeddings(source="corpus", golden_source="golden.public.golden", k=5)
for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
    value = metrics["aggregate"][key]
    assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"
    print(f"{key:<16} {value:.4f}")
assert len(metrics["per_query"]) > 0, "per_query must carry one record per query"

# %% [markdown]
# ## Triplets to train on
#
# `(anchor, positive, negative)` image triplets: for each image, the positive
# is the next image of its family and the negative an image of another
# family. What makes an image a "positive" — a redraw, another view of the same
# object, an augmentation — is the caller's data; the trainer only minimizes
# the triplet loss over whatever images are paired.

# %%
families: dict[str, list[bytes]] = {}
for path in paths:
    family = path.stem[len("img_"):].rsplit("_", 1)[0]
    families.setdefault(family, []).append(path.read_bytes())

names = list(families)
anchors, positives, negatives = [], [], []
for fi, name in enumerate(names):
    images, others = families[name], families[names[(fi + 1) % len(names)]]
    for ci, anchor in enumerate(images):
        anchors.append(anchor)
        positives.append(images[(ci + 1) % len(images)])
        negatives.append(others[ci % len(others)])

pq.write_table(
    pa.table(
        {
            "anchor": pa.array(anchors, type=pa.binary()),
            "positive": pa.array(positives, type=pa.binary()),
            "negative": pa.array(negatives, type=pa.binary()),
        }
    ),
    home / "image_triplets.parquet",
)
db.add_source("triplets", url=str(home / "image_triplets.parquet"), format="parquet")

# %% [markdown]
# ## Adapt the vision tower
#
# A non-empty `target_modules` puts LoRA inside the vision transformer itself:
# `in_proj` is its fused query-key-value projection and `c_fc` the MLP's first
# linear, so the tower's own representation moves. (The audio recipe shows the
# other mode: an empty list trains a projection head on a frozen tower.) The
# adapted model is registered under the image task, so `search` resolves it as
# an image encoder.

# %%
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
tuned_model = job.output_model_id
assert tuned_model.startswith("jammi:fine-tuned:"), f"unexpected model_id: {tuned_model}"
print(f"fine-tuned image model: {tuned_model}")

described = db.describe_model(tuned_model)
assert described is not None, f"{tuned_model} missing from the catalog"
assert described["task"] == "image_embedding", f"registered under the wrong task: {described}"

# %% [markdown]
# ## The adapter is served
#
# The same query image, encoded through the adapted model, must come out
# different from the base encoding: an adapter that trained but was dropped at
# serve time would leave the two vectors identical. This checks change, not
# improvement — the fixture's weights are random, so the direction of the
# change carries no information — and it checks the vectors rather than top-k
# metrics, which on a set this small rarely flip even when the vectors move.

# %%
tuned_query_vec = db.encode_query(model=tuned_model, query=query_png, modality="image")
assert len(tuned_query_vec) == len(query_vec), (
    f"tuned dim {len(tuned_query_vec)} differs from base dim {len(query_vec)}"
)
max_abs_diff = max(abs(b - t) for b, t in zip(query_vec, tuned_query_vec))
print(f"query embedding max |Δ| (base vs tuned): {max_abs_diff:.6f}")
assert max_abs_diff > 1e-4, (
    f"the tuned query vector equals the base one (max |Δ| = {max_abs_diff:.2e}): "
    "the adapter was trained but is not applied when the model is served"
)

# %% [markdown]
# ## A selector that matches nothing is refused
#
# `q_proj` is a real site on many decoder checkpoints and on nothing in an
# OpenCLIP tower. Selecting no site would train zero parameters and publish an
# adapter that changes nothing, so the engine fails the job instead, and the
# message names this tower's real sites.

# %%
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
except TrainingError as error:
    message = str(error)
    print(f"refused: {message}")
    assert "q_proj" in message, f"the refusal must echo the submitted selector: {message}"
    assert "in_proj" in message, f"the refusal must name this tower's real sites: {message}"
else:
    raise AssertionError("a target_modules list matching no site must fail the job")

# %%
db.close()
