"""Score a NER model's entity spans against gold spans: precision, recall, F1.

Run with `python cookbook/recipes/eval_inference_ner/example.py`, or a step at
a time as a notebook: each `# %%` cell is one step.
"""

# %%
import tempfile

import jammi
from jammi_cookbook import fixtures

MODEL = fixtures.model("tiny_modernbert_ner")

db = jammi.connect(f"file://{tempfile.mkdtemp()}")

# %% [markdown]
# ## Register the text and the gold spans
#
# The corpus is the text to tag. The gold source holds one row per entity
# span, `(id, label, start, end)`; spans sharing an `id` make up that row's
# gold set.

# %%
db.add_source("corpus", url=str(fixtures.path("tiny_ner_corpus.parquet")), format="parquet")
db.add_source("golden", url=str(fixtures.path("tiny_ner_gold.csv")), format="csv")

# %% [markdown]
# ## Run the model and score it
#
# `eval_inference` with `task="ner"` runs the model over `columns` and matches
# its predicted spans against the gold ones. Matching is strict at the entity
# level: a prediction is a true positive only when its `(label, start, end)`
# equals a gold span's.

# %%
metrics = db.eval_inference(
    model=MODEL,
    source="corpus",
    columns=["text"],
    task="ner",
    golden_source="golden.public.tiny_ner_gold",
    label_column="label",
)

aggregate = metrics["aggregate"]
assert aggregate["task"] == "ner", aggregate["task"]
for key in ("precision", "recall", "f1"):
    assert 0.0 <= aggregate[key] <= 1.0, f"{key} out of range: {aggregate[key]}"
print(f"precision: {aggregate['precision']:.4f}")
print(f"recall:    {aggregate['recall']:.4f}")
print(f"f1:        {aggregate['f1']:.4f}")

# %% [markdown]
# ## Read the breakdowns
#
# `aggregate["per_type"]` scores each entity type on its own. `per_record`
# holds one entry per row, with the predicted and the gold span lists side by
# side, which is where to look when a type scores low.

# %%
per_type = aggregate.get("per_type", {})
assert isinstance(per_type, dict), f"per_type shape: {type(per_type)}"
for label, stats in per_type.items():
    print(
        f"{label:<6} precision={stats['precision']:.4f}"
        f"  recall={stats['recall']:.4f}  f1={stats['f1']:.4f}"
        f"  support={stats['support']}"
    )

per_record = metrics["per_record"]
assert len(per_record) > 0, "per_record must carry one entry per aligned row"
for entry in per_record:
    assert entry["task"] == "ner", entry["task"]
    assert isinstance(entry["predicted"], list)
    assert isinstance(entry["gold"], list)
print(f"{len(per_record)} rows scored; the first: {per_record[0]}")

# %%
db.close()
