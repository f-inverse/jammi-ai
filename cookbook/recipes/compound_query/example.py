"""Compound retrieval and inference: join sources and run a model inside one SQL query.

Run with `python cookbook/recipes/compound_query/example.py`, or a step at a
time as a notebook: each `# %%` cell is one step. Needs a `jammi-server` on
PATH (`pip install jammi-server`).
"""

# %% [markdown]
# `search` is the bounded primitive; anything caller-shaped — joining sources,
# filtering, running a model over a relation — rides SQL. The `annotate(...)`
# table function runs a model over a relation's columns inside the query, so a
# join, a filter and model inference are one statement, planned by the engine.
# The same SQL runs in process and over the Flight SQL lane against a
# `jammi-server`.

# %%
import csv
import tempfile
from pathlib import Path

import numpy as np

import jammi
from jammi.testing import LiveServer
from jammi_cookbook import fixtures

CORPUS_URL = fixtures.url("tiny_corpus.parquet")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# %% [markdown]
# ## A second source to join
#
# The corpus is 20 patents, each with an `assignee_id` from 101 to 110. A small
# `assignees` table says who holds each patent and where.

# %%
ASSIGNEES = [
    (101, "Aurora Photonics", "US"),
    (102, "Brightwell Bio", "DE"),
    (103, "Cobalt Materials", "JP"),
    (104, "Delta Quantum", "US"),
    (105, "Ember Chemical", "DE"),
    (106, "Fjord Energy", "NO"),
    (107, "Granite Therapeutics", "US"),
    (108, "Harbor Robotics", "JP"),
    (109, "Iris Genomics", "US"),
    (110, "Juniper AI", "US"),
]

assignees_path = Path(tempfile.mkdtemp()) / "assignees.csv"
with assignees_path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "company_name", "country"])
    writer.writerows(ASSIGNEES)
assignees_url = assignees_path.as_uri()

# %% [markdown]
# ## Two statements
#
# The first enriches every patent with its assignee's company and country by a
# join. The second runs the model inside the query:
# `annotate(model, task, relation, key_column, content_column)` embeds every
# patent, and the statement joins each vector back to its patent and its
# assignee and keeps the US-held ones.

# %%
ENRICHED = """
    SELECT p.id, p.title, a.company_name, a.country
    FROM corpus.public.tiny_corpus AS p
    JOIN assignees.public.assignees AS a ON p.assignee_id = a.id
    ORDER BY p.id
"""

ANNOTATED = f"""
    SELECT p.id, p.title, a.company_name, ann.vector
    FROM annotate('{MODEL}', 'text_embedding',
                  'corpus.public.tiny_corpus', 'id', 'content') AS ann
    JOIN corpus.public.tiny_corpus  AS p ON ann._row_id = arrow_cast(p.id, 'Utf8')
    JOIN assignees.public.assignees AS a ON p.assignee_id = a.id
    WHERE a.country = 'US'
    ORDER BY p.id
"""


def compound(db) -> tuple[list[dict], list[dict]]:
    """Either transport: register both sources, then run both statements."""
    db.add_source("corpus", url=CORPUS_URL, format="parquet")
    db.add_source("assignees", url=assignees_url, format="csv")
    return db.sql(ENRICHED).to_pylist(), db.sql(ANNOTATED).to_pylist()


# %% [markdown]
# ## Run them in process
#
# Every patent gains its assignee's company and country. The annotated
# statement keeps exactly the US-held patents, each once, with a vector the
# model computed from that patent's text.

# %%
with jammi.connect(f"file://{tempfile.mkdtemp()}") as db:
    enriched, annotated = compound(db)

print(f"enriched: {len(enriched)} patents, e.g. {enriched[0]}")
assert len(enriched) == 20 and all(r["company_name"] for r in enriched)

us_ids = [r["id"] for r in enriched if r["country"] == "US"]
assert [r["id"] for r in annotated] == us_ids, "annotate kept every US patent once"
width = len(annotated[0]["vector"])
assert width > 0 and all(len(r["vector"]) == width for r in annotated)
print(f"annotated: {len(annotated)} US-held patents, {width}-dim vectors")
for row in annotated[:3]:
    print(f"  {row['id']:>2}  {row['company_name']:<20}  {row['title']}")

# %% [markdown]
# ## Aggregate vectors in SQL
#
# `vector_mean`, `vector_sum` and `vector_max` reduce a group of vectors
# element by element, the way `avg`, `sum` and `max` reduce numbers. Grouping
# the patents' vectors by category gives each category's centroid, the
# direction its abstracts share; searching the corpus with a centroid finds
# that category's patents. The sum is the count times the mean, and the
# element-wise maximum is at least the mean in every element.

# %%
CENTROIDS = f"""
    SELECT p.category, count(*) AS patents,
           vector_mean(ann.vector) AS centroid,
           vector_sum(ann.vector)  AS total,
           vector_max(ann.vector)  AS peak
    FROM annotate('{MODEL}', 'text_embedding',
                  'corpus.public.tiny_corpus', 'id', 'content') AS ann
    JOIN corpus.public.tiny_corpus AS p ON ann._row_id = arrow_cast(p.id, 'Utf8')
    GROUP BY p.category
    ORDER BY p.category
"""

with jammi.connect(f"file://{tempfile.mkdtemp()}") as db:
    db.add_source("corpus", url=CORPUS_URL, format="parquet")
    groups = db.sql(CENTROIDS).to_pylist()
    db.generate_embeddings(source="corpus", model=MODEL, columns=["content"], key="id")
    for g in groups:
        centroid, total, peak = (np.asarray(g[c]) for c in ("centroid", "total", "peak"))
        np.testing.assert_allclose(total, g["patents"] * centroid, rtol=1e-4, atol=1e-5)
        assert np.all(peak >= centroid - 1e-6)
        hits = db.search("corpus", query=g["centroid"], k=3,
                         select=["id", "category", "title"]).to_pylist()
        print(f"{g['category']:<12} ({g['patents']} patents) -> "
              f"{[(h['id'], h['category']) for h in hits]}")
        assert hits[0]["category"] == g["category"], (g["category"], hits)

# %% [markdown]
# ## Run them against a server
#
# The identical SQL, sent to a `jammi-server` over the Flight SQL lane,
# returns the same rows and the same vectors.

# %%
with tempfile.TemporaryDirectory() as srv_dir, LiveServer(srv_dir) as server:
    with jammi.connect(server.endpoint) as remote:
        remote_enriched, remote_annotated = compound(remote)

assert remote_enriched == enriched, "the join is the same on both transports"
assert [r["id"] for r in remote_annotated] == us_ids
for local, served in zip(annotated, remote_annotated):
    np.testing.assert_allclose(served["vector"], local["vector"], rtol=1e-5, atol=1e-6)
print(f"remote ({server.endpoint}): the same {len(remote_annotated)} rows and vectors")
