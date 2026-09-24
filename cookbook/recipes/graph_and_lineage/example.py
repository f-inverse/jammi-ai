"""Graphs over embeddings, and the lineage every result table carries.

A corpus's embeddings become a k-nearest-neighbour graph; the graph smooths
the embeddings (propagation) and, on its own, yields structure embeddings for
nodes with no text. Every one of those tables records what produced it, so the
engine can answer the provenance questions a pipeline needs:

1. `db.build_neighbor_graph` — the corpus's k-NN edges as a table
2. `db.propagate_embeddings` — features smoothed over the graph (a decoupled GNN)
3. `db.generate_structure_embeddings` — embeddings from the edges alone
4. `db.verify_materialization` — does the table still match its recorded definition?
5. `db.staleness` — is it fresh against the definition you expect now?
6. `db.derives_from` — which tables were built from this one?
7. `db.recompute` — re-run a table's recorded producer
8. `db.reconcile` — cross-check the catalog against the object store

Uses the local `cookbook/fixtures/tiny_bert` encoder; runs in seconds on CPU.

Run from the repo root:  python cookbook/recipes/graph_and_lineage/example.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAMMI_GPU__DEVICE", "-1")

import jammi

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURES = REPO_ROOT / "cookbook" / "fixtures"
CORPUS_PATH = FIXTURES / "tiny_corpus.parquet"
MODEL = f"local:{FIXTURES / 'tiny_bert'}"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp, jammi.connect(f"file://{tmp}") as db:
        db.add_source("corpus", url=str(CORPUS_PATH), format="parquet")
        emb = db.generate_embeddings(
            source="corpus", model=MODEL, columns=["content"], key="id", modality="text"
        )

        # 1. The k-NN graph over the corpus's embeddings. Graph verbs take the
        #    SOURCE id; the embedding table is found (or pinned) from it.
        graph = db.build_neighbor_graph("corpus", k=3, exact=True)
        edges = db.sql(f'SELECT COUNT(*) AS n FROM "jammi.{graph}"').to_pylist()[0]["n"]
        print(f"neighbor graph {graph}: {edges} edges")

        # 2. Propagation: each node's vector mixed with its neighbours' (one hop,
        #    alpha = how much of the node itself is kept). A new, searchable table.
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

        # 4. Every table records its definition. A deliberately wrong
        #    expectation returns the recorded one as `found`.
        recorded = db.verify_materialization(propagated, expected_definition="deadbeef")["found"]
        verdict = db.verify_materialization(propagated, expected_definition=recorded)
        print(f"verify against the recorded definition: {verdict['verdict']}")
        assert verdict["verdict"] in ("match", "match_with_unpinned_inputs")

        # 5. Staleness against the definition you expect now.
        print(f"staleness (same definition): {db.staleness(propagated, recorded)['staleness']}")
        stale = db.staleness(propagated, "0" * len(recorded))
        print(f"staleness (a changed definition): {stale['staleness']} "
              f"({', '.join(r['reason'] for r in stale['reasons'])})")

        # 6. Lineage: what was built from the base embeddings?
        for edge in db.derives_from(emb):
            print(f"derives_from {emb[:32]}…: {edge['kind']} → {edge['table'][:48]}…")

        # 7. Recompute the graph and report — without recomputing — what now
        #    depends on a stale input.
        report = db.recompute(graph, cascade="report_only")
        print(f"recomputed {len(report['recomputed'])} table(s); "
              f"downstream stale: {len(report['downstream_stale'])}")

        # 8. The catalog and the object store agree (a dry run reclaims nothing).
        drift = db.reconcile()
        print(f"reconcile: {drift}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
