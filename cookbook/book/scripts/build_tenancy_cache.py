#!/usr/bin/env python3
"""Emit the multi-tenant-as-a-measured-property cache (B2) — CPU, no GPU.

Tenant isolation in the engine is not a feature flag — it is a *measurable property*
of the catalog and the analyzer, and this vertical measures it. The behaviour is
transport-independent (catalog filtering + a TableScan rewrite), so it runs on the
embedded CPU ``Database`` against the committed ogbn-arxiv cache, with no GPU.

The true engine contract (the corrected KV-air model) has exactly two genuine
isolation layers plus one honest caveat:

* **catalog-listing isolation** — ``list_sources`` filters the registry to
  ``tenant_id = $cur OR IS NULL``; tenant A's listing excludes B's registration
  (a HARD zero leak).
* **row-level discriminator-column isolation** — the analyzer injects
  ``tenant_id = $cur OR IS NULL`` onto a ``TableScan`` *only when* the queried table
  carries a ``tenant_id`` column, so the SAME tagged source returns disjoint rows
  under A vs B (a HARD zero leak).
* **the caveat (a positive assertion)** — a discriminator-LESS source is GLOBALLY
  readable: A sees ALL of B's rows when it names the source. The engine does not
  authenticate; access-gating lives above it.

And one consequence worth measuring directly — **tenant-conditioned metric parity**:
the same retrieval recipe, run under two tenants over a discriminator-tagged source,
yields each tenant its OWN scoped result over a disjoint row partition. Same recipe,
two scopes, two answers — the engine scoping the data, not the recipe.

This reuses ``jammi_cookbook.rails`` verbatim (``tenant`` / ``assert_listing_isolated``
/ ``assert_rows_isolated``) — the corrected helpers, not a re-implementation. It does
NOT reintroduce the false "a separate source per tenant hides data" claim: a separate
per-tenant source is excluded from the LISTING, but its rows are globally readable if
named (the caveat).

Usage::

    python scripts/build_tenancy_cache.py            # embedded CPU engine
    python scripts/build_tenancy_cache.py --target file:///tmp/tn
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import jammi_ai
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import jammi_cookbook  # noqa: F401  # applies the determinism env on import
from jammi_cookbook import contracts
from jammi_cookbook.rails import assert_listing_isolated, assert_rows_isolated, tenant

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "tenancy_b"
RECALL_K = 10

# Opaque tenant UUIDs — the engine validates the form, never who the tenant is; the
# names carry no meaning. There IS a tenant; nothing about who.
TENANT_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
TENANT_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def _papers() -> list[dict]:
    return contracts.load_artifact("arxiv.papers").to_pylist()


def _embeddings() -> tuple[list[str], np.ndarray]:
    table = contracts.load_artifact("arxiv.embeddings")
    ids = [str(x) for x in table.column("_row_id").to_pylist()]
    vecs = np.asarray([list(v) for v in table.column("vector").to_pylist()], dtype=np.float32)
    return ids, vecs


def _golden() -> dict[str, set[str]]:
    relevant: dict[str, set[str]] = defaultdict(set)
    for r in contracts.load_artifact("arxiv.subject_golden").to_pylist():
        relevant[r["query_id"]].add(r["relevant_id"])
    return relevant


def emit(db, work: Path) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    papers = _papers()
    subject = {p["paper_id"]: p["subject"] for p in papers}
    subjects = sorted({p["subject"] for p in papers})
    # Partition the subjects between two tenants — a real, disjoint row partition over
    # the committed cache (not a synthetic toy): every paper belongs to exactly one
    # tenant by its subject class.
    a_subjects = set(subjects[::2])
    tenant_of = {pid: (TENANT_A if subject[pid] in a_subjects else TENANT_B)
                 for pid in subject}
    a_papers = sorted(pid for pid, t in tenant_of.items() if t == TENANT_A)
    b_papers = sorted(pid for pid, t in tenant_of.items() if t == TENANT_B)
    print(f"partition: tenant A {len(a_papers)} papers / tenant B {len(b_papers)} papers "
          f"(total {len(tenant_of)})", flush=True)

    def write_src(name: str, table: pa.Table) -> None:
        path = work / f"{name}.parquet"
        pq.write_table(table, path)
        db.add_source(name, url=str(path), format="parquet")

    # --- Property 1: catalog-listing isolation (HARD zero) --------------------
    print("\n[1] catalog-listing isolation", flush=True)
    with tenant(db, TENANT_A):
        write_src("papers_a", pa.table({"paper_id": a_papers}))
    with tenant(db, TENANT_B):
        write_src("papers_b", pa.table({"paper_id": b_papers}))
    with tenant(db, TENANT_A):
        a_listed = [s["source_id"] for s in db.list_sources()]
    assert_listing_isolated(a_listed, {"papers_b"}, tenant_id=TENANT_A)
    listing_leak = len(set(a_listed) & {"papers_b"})
    print(f"  tenant A lists {sorted(a_listed)}; B's 'papers_b' leaked: {listing_leak}",
          flush=True)

    # --- Property 2: discriminator-column row isolation (HARD zero) -----------
    print("\n[2] discriminator-column row isolation", flush=True)
    db.set_tenant("")  # register the shared tagged source as a global (tenant_id IS NULL)
    write_src("papers_tagged", pa.table({
        "paper_id": a_papers + b_papers,
        "tenant_id": [TENANT_A] * len(a_papers) + [TENANT_B] * len(b_papers),
    }))
    with tenant(db, TENANT_A):
        seen_a = [r["paper_id"] for r in db.sql(
            "SELECT paper_id FROM papers_tagged.public.papers_tagged").to_pylist()]
    assert_rows_isolated(seen_a, set(b_papers), tenant_id=TENANT_A)
    discriminator_leak = len(set(seen_a) & set(b_papers))
    print(f"  tenant A reads tagged source: {len(seen_a)} rows; B-tagged leaked: "
          f"{discriminator_leak}", flush=True)

    # --- Property 3: the caveat — a discriminator-less source is global -------
    print("\n[caveat] a discriminator-less source is globally readable", flush=True)
    with tenant(db, TENANT_B):
        write_src("papers_b_nodisc", pa.table({"paper_id": b_papers}))
    with tenant(db, TENANT_A):
        seen_global = [r["paper_id"] for r in db.sql(
            "SELECT paper_id FROM papers_b_nodisc.public.papers_b_nodisc").to_pylist()]
    caveat_visible = len(set(seen_global) & set(b_papers))
    print(f"  tenant A reads B's discriminator-less source: {caveat_visible} of "
          f"{len(b_papers)} B rows visible (by design — the engine does not authenticate)",
          flush=True)

    # --- Property 4: tenant-conditioned metric parity -------------------------
    print("\n[parity] the same recall recipe under each tenant, scoped by the engine",
          flush=True)
    relevant = _golden()
    ids, vecs = _embeddings()
    pos = {i: n for n, i in enumerate(ids)}

    def scoped_recall(tenant_id: str) -> tuple[float, int, int]:
        # Each tenant reads ONLY its own paper_ids from the discriminator-tagged source
        # — the engine enforces the scope; the recipe (cosine-kNN recall@10) is identical.
        with tenant(db, tenant_id):
            visible = {r["paper_id"] for r in db.sql(
                "SELECT paper_id FROM papers_tagged.public.papers_tagged").to_pylist()}
        local_ids = [i for i in ids if i in visible]
        sub = vecs[[pos[i] for i in local_ids]]
        norm = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-12)
        local_pos = {i: n for n, i in enumerate(local_ids)}
        scored = 0.0
        total = 0
        for qid in sorted(relevant):
            if qid not in local_pos:
                continue
            rel = {r for r in relevant[qid] if r in visible}
            if not rel:
                continue
            sims = norm @ norm[local_pos[qid]]
            sims[local_pos[qid]] = -np.inf
            top = np.argpartition(-sims, RECALL_K)[:RECALL_K]
            got = {local_ids[t] for t in top}
            scored += len(got & rel) / min(RECALL_K, len(rel))
            total += 1
        return scored / total if total else 0.0, len(visible), total

    a_recall, a_visible, a_queries = scoped_recall(TENANT_A)
    b_recall, b_visible, b_queries = scoped_recall(TENANT_B)
    print(f"  tenant A: recall@10 {a_recall:.4f}  over {a_visible} rows / {a_queries} queries",
          flush=True)
    print(f"  tenant B: recall@10 {b_recall:.4f}  over {b_visible} rows / {b_queries} queries",
          flush=True)
    # The partition is exact and disjoint: the two scopes tile the full cache.
    assert a_visible + b_visible == len(tenant_of), "the tenant partition must tile the cache"

    record = {
        "tenant_a": TENANT_A,
        "tenant_b": TENANT_B,
        "tenant_a_papers": len(a_papers),
        "tenant_b_papers": len(b_papers),
        "listing_leak": listing_leak,
        "discriminator_leak": discriminator_leak,
        "discriminator_rows_seen": len(seen_a),
        "caveat_visible": caveat_visible,
        "parity_a_recall_at_10": round(a_recall, 4),
        "parity_b_recall_at_10": round(b_recall, 4),
        "parity_a_visible": a_visible,
        "parity_b_visible": b_visible,
        "parity_a_queries": a_queries,
        "parity_b_queries": b_queries,
        "note": (
            "Tenant isolation measured live on CPU as a property of the engine. "
            "(1) Catalog-listing isolation: tenant A's list_sources excludes B's source "
            "(hard zero). (2) Discriminator-column row isolation: the analyzer injects "
            "tenant_id = $cur OR IS NULL onto a TableScan only when the table carries a "
            "tenant_id column, so the SAME tagged source returns disjoint rows under A "
            "vs B (hard zero). Caveat: a discriminator-LESS source is GLOBALLY READABLE "
            "— tenant A sees all of B's rows when it names the source (the engine does "
            "not authenticate). Parity: the same recall recipe under each tenant yields "
            "each its own scoped result over a disjoint row partition. No false "
            "'separate source hides data' claim: a separate source is hidden from the "
            "LISTING, not from a direct named read."
        ),
    }
    (ARTIFACTS / "tenancy_b.json").write_text(json.dumps(record, indent=2))

    metrics = {
        "listing_leak": {"value": float(listing_leak), "tol": 0.0},
        "discriminator_leak": {"value": float(discriminator_leak), "tol": 0.0},
        "caveat_visible": {"value": float(caveat_visible), "tol": 0.0},
        "parity_a_recall_at_10": {"value": round(a_recall, 3), "tol": 0.03},
        "parity_b_recall_at_10": {"value": round(b_recall, 3), "tol": 0.03},
    }
    (ARTIFACTS / "golden_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    _write_checksums()
    print("\n=== tenant isolation, measured ===", flush=True)
    print(f"  listing_leak={listing_leak}  discriminator_leak={discriminator_leak}  "
          f"caveat_visible={caveat_visible}", flush=True)
    print(f"  parity: A {a_recall:.4f} / B {b_recall:.4f} (disjoint scopes)", flush=True)
    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


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
