"""Audio-to-audio search with a CLAP model: index, search, evaluate, and two ways to adapt it.

Run with `python cookbook/recipes/audio_search/example.py`, or a step at a time
as a notebook: each `# %%` cell is one step.
"""

# %% [markdown]
# `JAMMI_AUDIO_MODEL` names the checkpoint, any CLAP audio model as a Hugging
# Face repo id or `local:<path>`; the default is the random-weight
# `htsat_clap_tiny` fixture, which runs offline in seconds.

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

AUDIO_CORPUS_DIR = fixtures.path("tiny_audio_corpus")
DEFAULT_MODEL = fixtures.model("htsat_clap_tiny")
MODEL = os.environ.get("JAMMI_AUDIO_MODEL", DEFAULT_MODEL)
METRICS = ("recall_at_k", "precision_at_k", "mrr", "ndcg")
print(f"model: {MODEL}")

home = Path(tempfile.mkdtemp())
db = jammi.connect(f"file://{home}")

# %% [markdown]
# ## Load the clips
#
# The corpus is 20 synthetic one-second clips in five timbre families, held
# inline as WAV bytes in a Parquet source: `clip_id`, and `audio`.

# %%
paths = sorted(AUDIO_CORPUS_DIR.glob("clip_*.wav"))
assert paths, f"no corpus clips under {AUDIO_CORPUS_DIR}"
pq.write_table(
    pa.table(
        {
            "clip_id": pa.array([p.stem for p in paths], type=pa.utf8()),
            "audio": pa.array([p.read_bytes() for p in paths], type=pa.binary()),
        }
    ),
    home / "corpus.parquet",
)
db.add_source("corpus", url=str(home / "corpus.parquet"), format="parquet")

# %% [markdown]
# ## Embed them
#
# `generate_embeddings` with `modality="audio"` runs the audio tower over the
# `audio` column. The encoder is read from the checkpoint's CLAP config and
# owns decoding, resampling and the log-mel front end; the vectors are
# L2-normalized.

# %%
db.generate_embeddings(
    source="corpus", model=MODEL, columns=["audio"], key="clip_id", modality="audio"
)

# %% [markdown]
# ## Search with a clip
#
# A query clip is encoded by the same tower, and `search` returns its nearest
# corpus clips by cosine similarity.

# %%
query_wav = (AUDIO_CORPUS_DIR / "queries" / "q_sine.wav").read_bytes()
query_vec = db.encode_query(model=MODEL, query=query_wav, modality="audio")
assert query_vec, "query embedding must be non-empty"
print(f"query embedding dim: {len(query_vec)}")

results = db.search("corpus", query=query_vec, k=5)
assert results.num_rows > 0, "search must return a non-empty top-K"
print(f"top-{results.num_rows} for q_sine: {results.column('clip_id').to_pylist()}")

# %% [markdown]
# ## Measure retrieval quality
#
# The golden set holds a held-out query clip per family and the corpus clips
# of that family. A `query_audio` (binary) column is what switches
# `eval_embeddings` to audio queries. The numbers are reported, not judged:
# the fixture's weights are random.

# %%
query_ids, query_audios, relevant_ids = [], [], []
for q in json.loads(fixtures.path("tiny_audio_golden.json").read_text()):
    audio_bytes = (AUDIO_CORPUS_DIR / q["query_audio"]).read_bytes()
    for rid in q["relevant_ids"]:
        query_ids.append(q["query_id"])
        query_audios.append(audio_bytes)
        relevant_ids.append(str(rid))
pq.write_table(
    pa.table(
        {
            "query_id": pa.array(query_ids, type=pa.utf8()),
            "query_audio": pa.array(query_audios, type=pa.binary()),
            "relevant_id": pa.array(relevant_ids, type=pa.utf8()),
        }
    ),
    home / "golden.parquet",
)
db.add_source("golden", url=str(home / "golden.parquet"), format="parquet")

base_metrics = db.eval_embeddings(source="corpus", golden_source="golden.public.golden", k=5)
for key in METRICS:
    value = base_metrics["aggregate"][key]
    assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"
    print(f"{key:<16} {value:.4f}")
assert len(base_metrics["per_query"]) > 0, "per_query must carry one record per query"

# %% [markdown]
# ## Triplets to train on
#
# `(anchor, positive, negative)` clip triplets: for each clip, the positive is
# the next clip of its family and the negative a clip of another family. What
# makes a clip a "positive" — augmentation-similar, or co-occurring — is the
# caller's data; the trainer only minimizes the triplet loss over whatever
# clips are paired.

# %%
families: dict[str, list[bytes]] = {}
for path in paths:
    family = path.stem[len("clip_"):].rsplit("_", 1)[0]
    families.setdefault(family, []).append(path.read_bytes())

names = list(families)
anchors, positives, negatives = [], [], []
for fi, name in enumerate(names):
    clips, others = families[name], families[names[(fi + 1) % len(names)]]
    for ci, anchor in enumerate(clips):
        anchors.append(anchor)
        positives.append(clips[(ci + 1) % len(clips)])
        negatives.append(others[ci % len(others)])

pq.write_table(
    pa.table(
        {
            "anchor": pa.array(anchors, type=pa.binary()),
            "positive": pa.array(positives, type=pa.binary()),
            "negative": pa.array(negatives, type=pa.binary()),
        }
    ),
    home / "audio_triplets.parquet",
)
db.add_source("triplets", url=str(home / "audio_triplets.parquet"), format="parquet")

# %% [markdown]
# ## Adapt it cheaply: a head on a frozen tower
#
# With no `target_modules`, fine-tuning trains a projection head on top of the
# frozen audio tower: cheap, and the tower is untouched. Re-embedding the
# corpus with the tuned model and evaluating again compares the two.

# %%
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
tuned_model = job.output_model_id
assert tuned_model.startswith("jammi:fine-tuned:"), f"unexpected model_id: {tuned_model}"
print(f"fine-tuned audio model: {tuned_model}")

db.generate_embeddings(
    source="corpus", model=tuned_model, columns=["audio"], key="clip_id", modality="audio"
)
tuned_metrics = db.eval_embeddings(source="corpus", golden_source="golden.public.golden", k=5)
for key in METRICS:
    value = tuned_metrics["aggregate"][key]
    assert 0.0 <= value <= 1.0, f"tuned {key} out of range: {value}"
for label, run in (("base ", base_metrics), ("tuned", tuned_metrics)):
    print(f"{label}: " + "  ".join(f"{k}={run['aggregate'][k]:.4f}" for k in METRICS))

# %% [markdown]
# ## The head changed the embedding
#
# The same query clip, encoded through the tuned model, must come out
# different from the base encoding. This checks the vectors, not the metrics
# above: on a set this small the rankings rarely flip even when the vectors
# move. It checks change, not improvement — the fixture's weights are random.

# %%
tuned_query_vec = db.encode_query(model=tuned_model, query=query_wav, modality="audio")
assert len(tuned_query_vec) == len(query_vec), (
    f"tuned dim {len(tuned_query_vec)} differs from base dim {len(query_vec)}"
)
max_abs_diff = max(abs(b - t) for b, t in zip(query_vec, tuned_query_vec))
print(f"query embedding max |Δ| (base vs tuned): {max_abs_diff:.6f}")
assert max_abs_diff > 1e-4, (
    f"the tuned query vector equals the base one (max |Δ| = {max_abs_diff:.2e}): "
    "the projection head did not change the embedding"
)

# %% [markdown]
# ## Adapt the tower itself
#
# A non-empty `target_modules` puts LoRA inside the HTSAT-Swin tower, on the
# same triplets: `query` and `value` are the Swin blocks' attention
# projections, `linear1` the audio projection's first linear. The tower's own
# representation moves — more capacity for a domain the base checkpoint never
# saw, at more compute. The adapted model registers under the audio task, and
# the same query encodes differently through it: an adapter that trained but
# was dropped at serve time would leave the two vectors identical.

# %%
tower_job = db.fine_tune(
    source="triplets",
    base_model=MODEL,
    columns=["anchor", "positive", "negative"],
    method="lora",
    task="audio_embedding",
    target_modules=["query", "value", "linear1"],
    lora_rank=4,
    learning_rate=5e-3,
    epochs=2,
    batch_size=4,
    warmup_steps=0,
    validation_fraction=0.0,
    early_stopping_metric="train_loss",
)
tower_job.wait()
tower_model = tower_job.output_model_id
assert tower_model.startswith("jammi:fine-tuned:"), f"unexpected model_id: {tower_model}"
print(f"tower-adapted audio model: {tower_model}")

described = db.describe_model(tower_model)
assert described is not None, f"{tower_model} missing from the catalog"
assert described["task"] == "audio_embedding", f"registered under the wrong task: {described}"

tower_query_vec = db.encode_query(model=tower_model, query=query_wav, modality="audio")
assert len(tower_query_vec) == len(query_vec), (
    f"tower-adapted dim {len(tower_query_vec)} differs from base dim {len(query_vec)}"
)
tower_max_abs_diff = max(abs(b - t) for b, t in zip(query_vec, tower_query_vec))
print(f"query embedding max |Δ| (base vs tower-adapted): {tower_max_abs_diff:.6f}")
assert tower_max_abs_diff > 1e-4, (
    f"the adapted query vector equals the base one (max |Δ| = {tower_max_abs_diff:.2e}): "
    "the adapter was trained but is not applied when the model is served"
)

# %% [markdown]
# ## A selector that matches nothing is refused
#
# `q_proj` is a real site on many decoder checkpoints and on nothing in an
# HTSAT-Swin tower. Selecting no site would train zero parameters and publish
# an adapter that changes nothing, so the engine fails the job, and the
# message names this tower's real sites. `linear1` appears only in this
# tower's site list, so it is the part of the message that proves it.

# %%
refused = db.fine_tune(
    source="triplets",
    base_model=MODEL,
    columns=["anchor", "positive", "negative"],
    method="lora",
    task="audio_embedding",
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
    assert "query" in message and "linear1" in message, (
        f"the refusal must name this tower's real sites: {message}"
    )
else:
    raise AssertionError("a target_modules list matching no site must fail the job")

# %%
db.close()
