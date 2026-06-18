#!/usr/bin/env python3
"""Emit the mutable-companion-table / feature-store cache (C1) — CPU, no GPU.

A mutable companion table is the engine's primitive for state you register *next to*
the append-only result tables and federate into queries: you provision it with a
schema and a primary key, populate it, and JOIN it into a query over a registered
source. This vertical builds a per-paper feature derived purely from the committed
ogbn-arxiv cache (each paper's **citation in-degree** over the committed declared
citation graph — a real graph-derived "how-cited-is-this-paper" feature, the kind a
feature store holds), commits it as a feature parquet, and records the measured
JOIN aggregate the chapter reproduces live against this golden.

The honest limit this vertical states plainly (and does NOT overclaim around): on the
embedded ``Database`` surface the mutable table is **append-only** — ``INSERT`` and
``SELECT`` round-trip, but ``UPDATE`` / ``DELETE`` / duplicate-key ``INSERT`` are not
wired through (they raise, or hit the primary-key UNIQUE constraint). In-place upsert
is a forthcoming engine capability not yet exposed on this surface. So the teaching is
*register → populate (append) → federate into a JOIN*, never "edit a feature in place
and watch the join change."

The feature is fully deterministic and reproducible: it reads ``arxiv.cite_edges`` and
``arxiv.papers`` from the committed cache and counts, for every paper in the subset,
how many committed citation edges point *to* it. The chapter re-derives the same
feature, INSERTs it into a live mutable table, and re-runs the subject-level
``SUM(in_degree)`` JOIN aggregate, asserting it against the goldens emitted here.

Usage::

    python scripts/build_feature_store_cache.py            # embedded CPU engine
    python scripts/build_feature_store_cache.py --target file:///tmp/fs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

import jammi_ai
import pyarrow as pa
import pyarrow.parquet as pq

import jammi_cookbook  # noqa: F401  # applies the determinism env on import
from jammi_cookbook import contracts

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "feature_store"

# The mutable table's schema + key — the same contract the chapter provisions.
FEATURE_SCHEMA = pa.schema([("paper_id", pa.string()), ("in_degree", pa.int64())])
PRIMARY_KEY = ["paper_id"]


def _papers() -> list[dict]:
    return contracts.load_artifact("arxiv.papers").to_pylist()


def _citation_in_degree(paper_ids: set[str]) -> dict[str, int]:
    """Each paper's in-degree over the committed declared citation graph.

    The feature: for every paper in the subset, how many committed ``cite_edges``
    point *to* it (``dst == paper_id``). A real graph-derived per-paper feature —
    "how cited is this paper, within the committed subset" — derived only from the
    committed cache, with no GPU and no recompute of any upstream embedding.
    """
    indeg: Counter[str] = Counter()
    for edge in contracts.load_artifact("arxiv.cite_edges").to_pylist():
        dst = edge["dst"]
        if dst in paper_ids:
            indeg[dst] += 1
    # Every paper carries a feature value (0 for an uncited paper) — a dense feature
    # column, not a sparse one, so the JOIN credits every paper.
    return {pid: indeg.get(pid, 0) for pid in paper_ids}


def emit(db, work: Path) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    papers = _papers()
    subject = {p["paper_id"]: p["subject"] for p in papers}
    paper_ids = set(subject)
    in_degree = _citation_in_degree(paper_ids)
    rows = sorted(in_degree)  # deterministic row order, keyed by paper_id

    feature_table = pa.table(
        {"paper_id": rows, "in_degree": [in_degree[pid] for pid in rows]},
        schema=FEATURE_SCHEMA,
    )
    (ARTIFACTS).mkdir(parents=True, exist_ok=True)
    pq.write_table(feature_table, ARTIFACTS / "paper_features.parquet")

    total_citations = int(sum(in_degree.values()))
    nonzero = int(sum(1 for v in in_degree.values() if v > 0))
    print(f"feature: citation in-degree over {len(rows)} papers "
          f"({nonzero} cited, total {total_citations})", flush=True)

    # --- Register the source + provision the mutable companion table ----------
    papers_path = work / "papers.parquet"
    pq.write_table(contracts.load_artifact("arxiv.papers"), papers_path)
    db.add_source("papers", url=str(papers_path), format="parquet")

    created = db.create_mutable_table(
        "paper_features", schema=FEATURE_SCHEMA, primary_key=PRIMARY_KEY)
    listed = db.list_mutable_tables()
    print(f"created mutable table {created!r}; list_mutable_tables -> "
          f"{[t['id'] for t in listed]}", flush=True)

    # --- Populate (append) the feature rows, then round-trip a few ------------
    values = ",".join(f"('{pid}', {in_degree[pid]})" for pid in rows)
    db.sql(f"INSERT INTO mutable.public.paper_features (paper_id, in_degree) VALUES {values}")
    roundtrip = db.sql(
        "SELECT paper_id, in_degree FROM mutable.public.paper_features "
        "ORDER BY in_degree DESC, paper_id LIMIT 3").to_pylist()
    populated = db.sql(
        "SELECT COUNT(*) AS n FROM mutable.public.paper_features").to_pylist()[0]["n"]
    print(f"populated {populated} feature rows; top by in_degree: "
          f"{[(r['paper_id'], r['in_degree']) for r in roundtrip]}", flush=True)

    # --- The honest append-only probe: UPDATE / DELETE / dup-INSERT all fail --
    append_only = _probe_append_only(db, rows[0])
    print(f"append-only probe: update={append_only['update_rejected']} "
          f"delete={append_only['delete_rejected']} "
          f"dup_insert={append_only['duplicate_insert_rejected']}", flush=True)

    # --- The measured aggregate: SUM(in_degree) per subject, via the JOIN -----
    agg = db.sql(
        "SELECT p.subject AS subject, SUM(f.in_degree) AS total "
        "FROM papers.public.papers p "
        "JOIN mutable.public.paper_features f ON p.paper_id = f.paper_id "
        "GROUP BY p.subject ORDER BY total DESC, subject"
    ).to_pylist()
    join_total = int(sum(r["total"] for r in agg))
    top = agg[0]
    top_subject = top["subject"]
    top_subject_total = int(top["total"])
    print(f"JOIN aggregate: {len(agg)} subjects, total in-degree {join_total}; "
          f"top {top_subject!r} = {top_subject_total}", flush=True)

    # The JOIN must conserve the feature mass: every paper joins exactly once.
    assert join_total == total_citations, (
        "the subject-level JOIN aggregate must sum to the total feature mass")

    record = {
        "feature": "citation_in_degree",
        "feature_source": "arxiv.cite_edges (declared citation graph)",
        "rows": len(rows),
        "cited_papers": nonzero,
        "total_citations": total_citations,
        "mutable_table": created,
        "primary_key": PRIMARY_KEY,
        "populated_rows": int(populated),
        "join_total_in_degree": join_total,
        "top_subject": top_subject,
        "top_subject_total": top_subject_total,
        "subject_totals": {r["subject"]: int(r["total"]) for r in agg},
        "append_only": append_only,
        "note": (
            "A mutable companion table provisioned alongside the append-only result "
            "tables, populated with a per-paper citation-in-degree feature derived from "
            "the committed declared citation graph, and federated into a JOIN over the "
            "papers source. The feature is graph-derived and deterministic. The table is "
            "APPEND-ONLY on this surface: INSERT and SELECT round-trip, but UPDATE / "
            "DELETE / duplicate-key INSERT are not wired through (they raise or hit the "
            "primary-key UNIQUE constraint). In-place upsert is a forthcoming engine "
            "capability not yet exposed on this surface — the teaching is register -> "
            "populate (append) -> federate, never edit-in-place. The measured verdict is "
            "the subject-level SUM(in_degree) JOIN aggregate (top subject + grand total), "
            "which the chapter reproduces live against this golden."
        ),
    }
    (ARTIFACTS / "feature_store.json").write_text(json.dumps(record, indent=2))

    metrics = {
        "total_in_degree": {"value": float(join_total), "tol": 0.0},
        "top_subject_total": {"value": float(top_subject_total), "tol": 0.0},
        "cited_papers": {"value": float(nonzero), "tol": 0.0},
        "populated_rows": {"value": float(populated), "tol": 0.0},
    }
    (ARTIFACTS / "golden_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True))

    _write_checksums()
    print("\n=== mutable companion table, measured ===", flush=True)
    print(f"  populated_rows={populated}  total_in_degree={join_total}  "
          f"top_subject={top_subject!r}({top_subject_total})", flush=True)
    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


def _probe_append_only(db, sample_paper_id: str) -> dict[str, bool]:
    """Confirm the surface is append-only: UPDATE / DELETE / dup-INSERT all fail.

    Each mutating op that is NOT a fresh-key append must raise — this is the honest
    limit the chapter measures, not narrates. ``True`` means the op was correctly
    rejected (the append-only property holds).
    """
    def rejected(stmt: str) -> bool:
        try:
            db.sql(stmt)
        except RuntimeError:
            return True
        return False

    return {
        "update_rejected": rejected(
            "UPDATE mutable.public.paper_features SET in_degree = 999 "
            f"WHERE paper_id = '{sample_paper_id}'"),
        "delete_rejected": rejected(
            "DELETE FROM mutable.public.paper_features "
            f"WHERE paper_id = '{sample_paper_id}'"),
        "duplicate_insert_rejected": rejected(
            "INSERT INTO mutable.public.paper_features (paper_id, in_degree) "
            f"VALUES ('{sample_paper_id}', 0)"),
    }


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _write_checksums() -> None:
    sums = {p.name: _checksum(p) for p in sorted(ARTIFACTS.glob("*"))
            if p.is_file() and p.name != "checksums.json"}
    (ARTIFACTS / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None,
                    help="connect() target — file:// for the embedded CPU engine "
                         "(a fresh temp catalog is used if omitted).")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as catalog, tempfile.TemporaryDirectory() as work:
        db = jammi_ai.connect(args.target or f"file://{catalog}")
        emit(db, Path(work))


if __name__ == "__main__":
    main()
