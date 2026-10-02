"""The model catalog: see what an engine serves and has trained, and clean it up.

Run with `python cookbook/recipes/model_catalog/example.py`, or a step at a
time as a notebook: each `# %%` cell is one step. Seconds on a CPU.
"""

# %%
import tempfile
from pathlib import Path

import jammi
from jammi.errors import BackendError
from jammi_cookbook import fixtures

BASE_MODEL = fixtures.model("tiny_bert")

home = Path(tempfile.mkdtemp())
engine = f"file://{home}/engine"
db = jammi.connect(engine)

# %% [markdown]
# ## What this engine is
#
# `get_server_info` reports the engine's version, its features and the storage
# backends it can write results to. `describe_source` reports one registered
# source and the result tables built on it, and `None` for a name that is not
# registered.

# %%
info = db.get_server_info()
print(f"engine {info['version']}; storage backends: {info['storage_backends']}")

db.add_source("training", url=str(fixtures.path("tiny_pairs.csv")), format="csv")
source = db.describe_source("training")
print(f"source {source['source_id']}: {source['status']}")
assert db.describe_source("no_such_source") is None

# %% [markdown]
# ## Load a model before it is needed
#
# `preload_model` pays a model's load into the cache now, so the first request
# that uses it does not.

# %%
db.preload_model(BASE_MODEL)

# %% [markdown]
# ## Train a model into the catalog
#
# Training is what puts models in the catalog: the base model is registered
# when the job is submitted, the fine-tuned one when it completes. This is a
# one-epoch LoRA over 30 text pairs. `list_models` lists every model the
# catalog holds, and `describe_model` one of them.

# %%
job = db.fine_tune(
    source="training",
    base_model=BASE_MODEL,
    columns=["text_a", "text_b", "score"],
    method="lora",
    task="text_embedding",
    lora_rank=4,
    epochs=1,
)
job.wait()
tuned = job.output_model_id

for model in db.list_models():
    print(f"model {model['model_id'][-40:]}: {model['task']} ({model['status']})")
described = db.describe_model(tuned)
print(f"describe_model: {described}")
assert described is not None and described["model_id"] == tuned

# %% [markdown]
# ## Why a delete is refused
#
# A finished job keeps the models it references for `[jobs] retention_days`
# (30 by default) after it ends, so `delete_model` refuses the model the job
# just trained. Deleting a model that does not exist is a no-op with
# `if_exists=True`.

# %%
try:
    db.delete_model(tuned)
    raise AssertionError("a referenced model must not delete")
except BackendError as refused:
    print(f"delete while referenced: refused ({refused})")

db.delete_model("jammi:fine-tuned:absent", if_exists=True)
db.close()

# %% [markdown]
# ## Delete it once nothing references it
#
# Retention is deployment configuration. Reopened with a zero-day retention,
# the finished job is past it: the engine sweeps such jobs when it opens, and
# `prune_jobs` sweeps them on demand. With the job gone, the model deletes.

# %%
config = home / "jammi.toml"
config.write_text("[jobs]\nretention_days = 0\n")

db = jammi.connect(engine, config=str(config))
db.prune_jobs()
assert db.list_jobs() == [], "the finished job is gone"
db.delete_model(tuned)
assert db.describe_model(tuned) is None
print(f"deleted {tuned}")

# %%
db.close()
