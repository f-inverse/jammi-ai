"""Graphs over embeddings, and the lineage every result table carries.

A corpus's embeddings become a k-nearest-neighbour graph; the graph smooths
the embeddings (propagation) and, on its own, yields structure embeddings for
nodes with no text. Every one of those tables records what produced it, so the
engine answers the provenance questions a pipeline asks:

1. `db.build_neighbor_graph` — the corpus's k-NN edges as a table
2. `db.propagate_embeddings` — features smoothed over the graph (a decoupled GNN)
3. `db.generate_structure_embeddings` — embeddings from the edges alone
4. `db.describe_table` — what produced a table: descriptor, models, inputs
5. `db.verify_materialization` — do the bytes still match that record?
6. `db.staleness` — is the table still the output of the definition you hold?
7. `db.derives_from` — which tables were built from this one?
8. `db.recompute` — re-run a table's recorded producer
9. `db.reconcile` — cross-check the catalog against the object store

Uses the `tiny_bert` fixture encoder; runs in seconds on CPU.

Run with `python cookbook/recipes/graph_and_lineage/example.py`.
"""

from __future__ import annotations

import tempfile

import jammi
from jammi_cookbook import fixtures

CORPUS_PATH = fixtures.path("tiny_corpus.parquet")
MODEL = fixtures.model("tiny_bert")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp, jammi.connect(f"file://{tmp}") as db:
        db.add_source("corpus", url=str(CORPUS_PATH), format="parquet")
        emb = db.generate_embeddings(
            source="corpus", model=MODEL, columns=["content"], key="id", modality="text"
        )

        # 1. The k-NN graph over the corpus's embeddings. Graph verbs take the
        #    SOURCE id; the embedding table is the source's (or pass `table=`).
        graph = db.build_neighbor_graph("corpus", k=3, exact=True)
        edges = db.sql(f'SELECT COUNT(*) AS n FROM "jammi.{graph}"').to_pylist()[0]["n"]
        print(f"neighbor graph: {edges} edges (20 rows x k=3)")
        assert edges == 60

        # 2. Propagation: each node's vector mixed with its neighbours' (one hop;
        #    `alpha` is how much of the node itself is kept). A new, searchable
        #    embedding table.
        propagated = db.propagate_embeddings(
            "corpus", embedding_table=emb, edge_graph_table=graph, hops=1, alpha=0.5
        )
        print(f"propagated embeddings: {propagated}")

        # 3. Structure embeddings: vectors from the graph's shape alone, for
        #    nodes that carry no content to embed.
        structure = db.generate_structure_embeddings(
            "corpus", key_column="id", edge_graph_table=graph, dimensions=8, seed=0
        )
        print(f"structure embeddings: {structure}")

        # 4. What produced the propagated table: the producing verb and its
        #    parameters, the models that ran (none: propagation runs no model),
        #    and the tables it read, each anchored by digest.
        described = db.describe_table(propagated)
        recorded = described["definition_hash"]
        print(f"produced by: {described['descriptor']['producer']}")
        print(f"models run: {len(described['env']['models'])}")
        for anchor in described["input_anchors"]:
            print(f"  input {anchor['source'][:40]}… anchored by {anchor['kind']}")
        assert described["env"]["models"] == []

        # 5. The bytes still hash to the recorded digest, and the definition you
        #    expect is the one recorded.
        verdict = db.verify_materialization(propagated, expected_definition=recorded)
        print(f"verify against the recorded definition: {verdict['verdict']}")
        assert verdict["verdict"] == "match"

        # 6. Staleness against a definition: the recorded one is fresh (every
        #    input is another table, pinned by digest); a different one is stale
        #    and says why.
        fresh = db.staleness(propagated, recorded)
        print(f"staleness (recorded definition): {fresh['staleness']}")
        assert fresh["staleness"] == "fresh"
        stale = db.staleness(propagated, "0" * len(recorded))
        reasons = ", ".join(r["reason"] for r in stale["reasons"])
        print(f"staleness (another definition): {stale['staleness']} ({reasons})")
        assert stale["staleness"] == "stale"

        # 7. Lineage: the tables built from the base embeddings.
        derived = db.derives_from(emb)
        for edge in derived:
            print(f"derives_from: {edge['derived'][:48]}… ({edge['kind']})")
        assert {graph, propagated} <= {edge["derived"] for edge in derived}

        # 8. Recompute the graph from its recorded descriptor, and report —
        #    without recomputing — what depends on it.
        report = db.recompute(graph, cascade="report_only")
        print(
            f"recomputed {len(report['recomputed'])} table(s); "
            f"downstream stale: {len(report['downstream_stale'])}"
        )
        assert len(report["recomputed"]) == 1

        # 9. The catalog and the object store agree. A dry run reclaims nothing.
        drift = db.reconcile()
        print(f"reconcile (dry run): {drift}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
