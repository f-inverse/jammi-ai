"""Measure recall@k, precision@k, MRR and nDCG of an embedding table against a golden set.

Run with `python cookbook/recipes/eval_embeddings/example.py`, or a step at a
time as a notebook: each `# %%` cell is one step.
"""

# %%
import csv
import json
import tempfile
from pathlib import Path

import jammi
from jammi_cookbook import fixtures

MODEL = fixtures.model("tiny_bert")

home = Path(tempfile.mkdtemp())
db = jammi.connect(f"file://{home}")

# %% [markdown]
# ## Embed the corpus
#
# The table under evaluation: 32-dimensional embeddings of the corpus's
# `content`.

# %%
db.add_source("corpus", url=str(fixtures.path("tiny_corpus.parquet")), format="parquet")
base = db.generate_embeddings(
    source="corpus", model=MODEL, columns=["content"], key="id", modality="text"
)

# %% [markdown]
# ## The golden set
#
# Relevance judgments: for each query, the corpus rows that should come back.
# `eval_embeddings` reads them as a source with one row per
# `(query_id, query_text, relevant_id)`, so the committed JSON, one entry per
# query with a list of relevant ids, is flattened into that shape first.

# %%
golden_csv = home / "golden.csv"
with golden_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["query_id", "query_text", "relevant_id"])
    for q in json.loads(fixtures.path("tiny_golden.json").read_text()):
        for rid in q["relevant_ids"]:
            writer.writerow([q["query_id"], q["query_text"], str(rid)])

db.add_source("golden", url=str(golden_csv), format="csv")

# %% [markdown]
# ## Evaluate
#
# `eval_embeddings` encodes every query with the table's own model, searches
# the source's embedding table to depth `k`, and scores the ranking against
# the judgments. `aggregate` holds the means over the queries; `per_query`
# holds each query's own scores.

# %%
metrics = db.eval_embeddings(source="corpus", golden_source="golden.public.golden", k=5)

aggregate = metrics["aggregate"]
for key in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
    assert 0.0 <= aggregate[key] <= 1.0, f"{key} out of range: {aggregate[key]}"
    print(f"{key:<16} {aggregate[key]:.4f}")

per_query = metrics["per_query"]
assert len(per_query) > 0, "per_query must carry one record per query"
assert per_query[0]["query_id"], "per_query records carry the golden-set query_id"
print(f"per_query: {len(per_query)} records (first: {per_query[0]['query_id']})")

# %% [markdown]
# ## The run is kept
#
# Each run's per-query results are stored in the catalog under its
# `eval_run_id`, with recall at 1, 3, 5 and 10, MRR, nDCG and distance for
# each query. `eval_per_query` reads them back.

# %%
eval_run_id = metrics["eval_run_id"]
persisted = db.eval_per_query(eval_run_id)
assert len(persisted) == len(per_query), "one persisted row per query"
for key in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg", "distance"):
    assert key in persisted[0]["metrics"], f"persisted metric '{key}' present"
print(f"persisted per-query rows: {len(persisted)} (run {eval_run_id})")

# %% [markdown]
# ## Tag queries by cohort
#
# `cohorts` attaches tags to queries by id, so quality can later be broken
# down by segment. The engine stores them as given and never interprets them.

# %%
tagged = db.eval_embeddings(
    source="corpus", golden_source="golden.public.golden", k=5, cohorts={"q1": {"split": "val"}}
)
by_query = {r["query_id"]: r for r in db.eval_per_query(tagged["eval_run_id"])}
assert by_query["q1"]["cohorts"].get("split") == "val", "cohort stored verbatim"
print(f"q1 cohorts: {by_query['q1']['cohorts']}")

# %% [markdown]
# ## Compare two tables
#
# `eval_compare` scores several embedding tables on one golden set. The first
# is the baseline, and every other carries its per-metric delta against it.
# Here the candidate is the same embeddings smoothed over the corpus's own
# k-nearest-neighbour graph.

# %%
graph = db.build_neighbor_graph("corpus", k=3, exact=True)
smoothed = db.propagate_embeddings(
    "corpus", embedding_table=base, edge_graph_table=graph, hops=1, alpha=0.5
)
compared = db.eval_compare(
    embedding_tables=[base, smoothed], source="corpus", golden_source="golden.public.golden", k=5
)
baseline, candidate = compared["per_table"]
assert baseline["delta"] is None
base_recall = baseline["embedding_eval"]["aggregate"]["recall_at_k"]
recall_delta = candidate["delta"]["recall_at_k"]["absolute"]
print(f"base recall@5 {base_recall:.4f}; smoothed Δ {recall_delta:+.4f}")

# %%
db.close()
