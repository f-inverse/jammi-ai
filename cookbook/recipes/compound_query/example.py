"""Compound retrieval and inference: join sources and run a model inside one SQL query.

`search` is the bounded primitive; anything caller-shaped — joining sources,
filtering, running a model over a relation — rides SQL. The `annotate(...)`
table function runs a model over a relation's columns inside the query, so a
join, a filter and model inference are one statement, planned by the engine.
The same SQL runs in process on the embedded engine and over the Flight SQL
lane against a `jammi-server`.

1. Register a patent corpus and an `assignees` table (who holds each patent)
2. Enrich by joining: each patent's company and country, in SQL
3. Annotate: `annotate(model, task, relation, key_column, content_column)`
   embeds every patent inside the query, joined back to its row and its
   assignee, filtered to US assignees — one statement
4. Run the identical SQL against a `jammi-server` over Flight SQL, and check
   it returns the same rows and the same vectors

Needs a `jammi-server` binary on PATH: `pip install jammi-server`, or
`cargo build --release -p jammi-server` with `target/release` on PATH.

Run with `python cookbook/recipes/compound_query/example.py`.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import jammi
import numpy as np
from jammi.testing import LiveServer
from jammi_cookbook import fixtures

CORPUS_URL = fixtures.url("tiny_corpus.parquet")
MODEL = fixtures.model("tiny_bert")

# Who holds each patent: the corpus's `assignee_id` values, 101–110.
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

ENRICHED = """
    SELECT p.id, p.title, a.company_name, a.country
    FROM corpus.public.tiny_corpus AS p
    JOIN assignees.public.assignees AS a ON p.assignee_id = a.id
    ORDER BY p.id
"""

# Search → join → annotate in one statement: the model runs inside the query.
ANNOTATED = f"""
    SELECT p.id, p.title, a.company_name, ann.vector
    FROM annotate('{MODEL}', 'text_embedding',
                  'corpus.public.tiny_corpus', 'id', 'content') AS ann
    JOIN corpus.public.tiny_corpus  AS p ON ann._row_id = arrow_cast(p.id, 'Utf8')
    JOIN assignees.public.assignees AS a ON p.assignee_id = a.id
    WHERE a.country = 'US'
    ORDER BY p.id
"""


def write_assignees(directory: Path) -> str:
    """The `assignees` table as a CSV file, returned as its `file://` URL."""
    path = directory / "assignees.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "company_name", "country"])
        writer.writerows(ASSIGNEES)
    return path.as_uri()


def compound(db, assignees_url: str) -> tuple[list[dict], list[dict]]:
    """The same program on either transport: the enriched rows, then the annotated ones."""
    db.add_source("corpus", url=CORPUS_URL, format="parquet")
    db.add_source("assignees", url=assignees_url, format="csv")
    return db.sql(ENRICHED).to_pylist(), db.sql(ANNOTATED).to_pylist()


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        assignees_url = write_assignees(Path(scratch))

        with tempfile.TemporaryDirectory() as local_dir, jammi.connect(f"file://{local_dir}") as db:
            enriched, annotated = compound(db, assignees_url)

        # Every patent gains its assignee's company and country from the join.
        print(f"enriched: {len(enriched)} patents, e.g. {enriched[0]}")
        assert len(enriched) == 20 and all(r["company_name"] for r in enriched)

        # The annotated query keeps exactly the US-held patents — each once —
        # with a vector the model computed from that patent's text.
        us_ids = [r["id"] for r in enriched if r["country"] == "US"]
        assert [r["id"] for r in annotated] == us_ids, "annotate kept every US patent once"
        width = len(annotated[0]["vector"])
        assert width > 0 and all(len(r["vector"]) == width for r in annotated)
        print(f"annotated: {len(annotated)} US-held patents, {width}-dim vectors")
        for row in annotated[:3]:
            print(f"  {row['id']:>2}  {row['company_name']:<20}  {row['title']}")

        # The identical SQL against a server, over the Flight SQL lane.
        with tempfile.TemporaryDirectory() as srv_dir, LiveServer(srv_dir) as server:
            with jammi.connect(server.endpoint) as remote:
                remote_enriched, remote_annotated = compound(remote, assignees_url)

        assert remote_enriched == enriched, "the join is the same on both transports"
        assert [r["id"] for r in remote_annotated] == us_ids
        for local, served in zip(annotated, remote_annotated):
            np.testing.assert_allclose(served["vector"], local["vector"], rtol=1e-5, atol=1e-6)
        print(f"remote ({server.endpoint}): the same {len(remote_annotated)} rows and vectors")

    print("compound_query: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
