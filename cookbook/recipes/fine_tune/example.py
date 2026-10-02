"""Fine-tune `tiny_bert` with LoRA, from labelled pairs and from a citation graph.

Run with `python cookbook/recipes/fine_tune/example.py`, or a step at a time
as a notebook: each `# %%` cell is one step. Seconds on a CPU.
"""

# %%
import tempfile

import jammi
from jammi_cookbook import fixtures

BASE_MODEL = fixtures.model("tiny_bert")

db = jammi.connect(f"file://{tempfile.mkdtemp()}")

# %% [markdown]
# ## Register the training pairs
#
# Thirty contrastive pairs, `(text_a, text_b, score)`: how similar the two
# texts should come out.

# %%
db.add_source("training", url=str(fixtures.path("tiny_pairs.csv")), format="csv")

# %% [markdown]
# ## Submit the fine-tune
#
# `fine_tune` submits a job and returns its handle at once; `wait` blocks
# until the job finishes. The defaults suit production workloads; this keeps
# the LoRA rank small and runs one epoch so it finishes in seconds. The
# trained model is registered in the catalog under a `jammi:fine-tuned:` id.

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
assert job.job_id, "fine_tune returned a job without an id"
print(f"job_id:   {job.job_id}")

job.wait()
model_id = job.output_model_id
assert model_id.startswith("jammi:fine-tuned:"), f"unexpected model_id: {model_id}"
print(f"model_id: {model_id}")

# %% [markdown]
# ## Use the fine-tuned model
#
# The new id names a model like any other: here it encodes a query, loaded
# from the catalog, in the base model's 32 dimensions.

# %%
query_vec = db.encode_query(model=model_id, query="quantum computing applications")
assert len(query_vec) == 32, f"tiny_bert is 32-dim; got {len(query_vec)}-dim from fine-tuned"

# %% [markdown]
# ## Fine-tune from a graph instead
#
# With no labelled pairs, a graph supplies them: `fine_tune_graph` pulls
# together papers that cite each other, sampling positives by random walks
# over the citation edges and negatives from the graph's non-neighbours. The
# nodes and the edges are two ordinary sources.

# %%
citations = fixtures.path("tiny_citation_graph")
db.add_source("papers", url=str(citations / "nodes.jsonl"), format="jsonl")
db.add_source("cites", url=str(citations / "edges.jsonl"), format="jsonl")

graph_job = db.fine_tune_graph(
    node_source="papers",
    id_column="id",
    text_column="text",
    edge_source="cites",
    base_model=BASE_MODEL,
    lora_rank=4,
    epochs=1,
    sample_seed=0,
)
graph_job.wait()
assert graph_job.output_model_id.startswith("jammi:fine-tuned:")
print(f"graph-tuned model_id: {graph_job.output_model_id}")

# %%
db.close()
