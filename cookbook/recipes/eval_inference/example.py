"""Score a classifier against gold labels: accuracy, macro F1, per class.

Run with `python cookbook/recipes/eval_inference/example.py`, or a step at a
time as a notebook: each `# %%` cell is one step.
"""

# %%
import tempfile

import jammi
from jammi_cookbook import fixtures

MODEL = fixtures.model("tiny_modernbert_classifier")

db = jammi.connect(f"file://{tempfile.mkdtemp()}")

# %% [markdown]
# ## Register the corpus and the gold labels
#
# The gold source is one `(id, label)` row per labelled corpus row.

# %%
db.add_source("corpus", url=str(fixtures.path("tiny_corpus.parquet")), format="parquet")
db.add_source("golden", url=str(fixtures.path("tiny_labels.csv")), format="csv")

# %% [markdown]
# ## Run the classifier and score it
#
# `eval_inference` runs the model over `columns`, aligns each prediction with
# its gold label by `id`, and returns the scores: `aggregate` holds accuracy
# and macro F1 (F1 averaged over the classes), tagged with the task.

# %%
metrics = db.eval_inference(
    model=MODEL,
    source="corpus",
    columns=["content"],
    task="classification",
    golden_source="golden.public.tiny_labels",
    label_column="label",
)

aggregate = metrics["aggregate"]
assert aggregate["task"] == "classification", aggregate["task"]
for key in ("accuracy", "f1"):
    assert 0.0 <= aggregate[key] <= 1.0, f"{key} out of range: {aggregate[key]}"
print(f"accuracy:  {aggregate['accuracy']:.4f}")
print(f"macro_f1:  {aggregate['f1']:.4f}")

# %% [markdown]
# ## Read the breakdowns
#
# `aggregate["per_class"]` scores each label on its own; `per_record` holds one
# aligned prediction and gold label per row.

# %%
per_class = aggregate.get("per_class", {})
assert isinstance(per_class, dict), f"per_class shape: {type(per_class)}"
for label, stats in per_class.items():
    print(
        f"{label:<12} precision={stats['precision']:.4f}"
        f"  recall={stats['recall']:.4f}  f1={stats['f1']:.4f}"
    )

per_record = metrics["per_record"]
assert len(per_record) > 0, "per_record must carry one entry per aligned row"
print(f"{len(per_record)} rows scored; the first: {per_record[0]}")

# %% [markdown]
# ## The predictions alone
#
# Without gold labels, `infer` runs the model over a source and returns one
# row per source row, keyed by `key`, with the model's outputs as columns.

# %%
predictions = db.infer(
    source="corpus",
    model=MODEL,
    columns=["content"],
    task="classification",
    key="id",
)
assert predictions.num_rows == 20, predictions.num_rows
print(f"{predictions.num_rows} rows, columns {predictions.column_names}")

# %%
db.close()
