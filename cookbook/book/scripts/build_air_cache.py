#!/usr/bin/env python3
"""Emit the Air Routes on-ramp cache (tiers 01–02) — run ONCE on the GPU server.

Air Routes is Neptune's own teaching dataset: 3504 airports, a declared
``route`` graph and a ``contains`` continent→country→airport hierarchy. It is
small enough to run *exactly* and fast — the clean 01–02 on-ramp. The heavy
pipeline (embed → neighbor graph → propagate) runs the real ``jammi_ai`` API
end-to-end on the committed airport subset and writes ``artifacts/air/*`` +
``golden_metrics.json``; every chapter (and CI) READS that cache on CPU and never
recomputes it.

This is the SHORT vertical: it stops at tier 02. Air Routes' text is too thin and
its label (continent ≈ solved by lat/lon) too near-deterministic for a credible
tier-03/04 — those live on ogbn-arxiv (the keystone). No context predictor, no
conformal story here.

The GPU compute tier cannot run on the CPU embed wheel: this script connects to a
running ``jammi-server`` (``jammi-server-cu12``) over ``grpc://`` and the engine
does the embedding / propagation on the device. The committed cache is then read
on CPU by the chapters via ``connect("file://…")`` — the ``connect(target)``
parity spine.

Usage::

    # 1. start the GPU server (clean artifact dir for a reproducible emit):
    #    JAMMI_ARTIFACT_DIR=/tmp/srv JAMMI_GPU__DEVICE=0 JAMMI_GPU__REQUIRE_GPU=true \\
    #    JAMMI_SERVER__FLIGHT_LISTEN=127.0.0.1:50051 jammi-server &
    # 2. emit against it:
    python scripts/build_air_cache.py --target grpc://127.0.0.1:50051
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jammi_ai
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import jammi_cookbook  # noqa: F401  # applies the determinism env on import
from jammi_cookbook import datasets

EMBED_MODEL = "answerdotai/ModernBERT-base"
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "air"
NEIGHBOR_K = 10
PROP_HOPS = 2
RECALL_K = 10


# --------------------------------------------------------------------------- #
# helpers (mirrors build_arxiv_cache.py)
# --------------------------------------------------------------------------- #


def _emb_ref(table: str) -> str:
    """An engine-produced table is addressed as a single quoted identifier."""
    return f'"jammi.{table}"'


def _read_vectors(db, table: str) -> tuple[list[str], np.ndarray]:
    """Read an embedding table's ``(_row_id, vector)`` as ids + an f32 matrix."""
    t = db.sql(f"SELECT _row_id, vector FROM {_emb_ref(table)} ORDER BY _row_id")
    ids = [str(x) for x in t.column("_row_id").to_pylist()]
    vecs = np.asarray([list(v) for v in t.column("vector").to_pylist()], dtype=np.float32)
    return ids, vecs


def _dump_emb(db, table: str, dest: Path) -> None:
    pq.write_table(db.sql(f"SELECT _row_id, vector FROM {_emb_ref(table)}"), dest)


def _dump_table(db, table: str, dest: Path) -> None:
    pq.write_table(db.sql(f"SELECT * FROM {_emb_ref(table)}"), dest)


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# measurement: same-continent retrieval recall@k (a label target, not similarity)
# --------------------------------------------------------------------------- #


def build_continent_golden(airport_rows: list[dict], *, query_n: int) -> tuple[pa.Table, str]:
    """A same-continent retrieval golden: query airport → same-continent airports.

    The relevance target is the continent label, independent of the embedding
    similarity — so route-graph propagation improving same-continent retrieval is a
    real low-pass-denoising result, not a circular one. (Continent is near-solved by
    lat/lon, so this is honestly a *teaching* label, not a benchmark — but it is a
    legitimate target for the raw→propagated delta.)
    """
    by_cont: dict[str, list[str]] = {}
    for r in airport_rows:
        by_cont.setdefault(r["continent"], []).append(r["code"])
    queries = [r for r in sorted(airport_rows, key=lambda r: r["code"])
               if r["continent"] and len(by_cont[r["continent"]]) >= 5][:query_n]
    rows = []
    for q in queries:
        for code in by_cont[q["continent"]]:
            if code != q["code"]:
                rows.append({"query_id": q["code"], "relevant_id": code})
    table = pa.table({
        "query_id": [r["query_id"] for r in rows],
        "relevant_id": [r["relevant_id"] for r in rows],
    })
    path = ARTIFACTS / "continent_golden.parquet"
    pq.write_table(table, path)
    return table, str(path)


def recall_at_k(db, *, emb_table: str, golden_rows: list[dict],
                query_ids: list[str], k: int) -> float:
    """recall@k for same-continent retrieval over a SPECIFIC embedding table.

    Exact cosine-similarity kNN over the committed embedding matrix (a pure numpy
    fold — deterministic, and targets exactly this table). ``search`` targets a
    *source*, which is ambiguous once a source has several embedding tables (raw,
    propagated); resolving the table by name here is the correct per-table
    comparison — the keystone's proven measurement.
    """
    relevant: dict[str, set[str]] = {}
    for r in golden_rows:
        relevant.setdefault(r["query_id"], set()).add(r["relevant_id"])
    ids, vecs = _read_vectors(db, emb_table)
    pos = {i: n for n, i in enumerate(ids)}
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    scored = 0.0
    total = 0
    for qid in query_ids:
        if qid not in pos or qid not in relevant:
            continue
        sims = norm @ norm[pos[qid]]
        sims[pos[qid]] = -np.inf  # exclude self
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        got = [ids[t] for t in top]
        rel = relevant[qid]
        scored += sum(1 for g in got if g in rel) / min(k, len(rel))
        total += 1
    return scored / total if total else 0.0


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


# Tenant ids are opaque UUIDs (the engine validates the form); the engine knows
# there *is* a tenant, nothing about who — the names below are arbitrary.
TENANT_A = "11111111-1111-1111-1111-111111111111"  # North America
TENANT_B = "22222222-2222-2222-2222-222222222222"  # Europe


def emit(db, target: str) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    info = db.get_server_info()
    print("server:", json.dumps(info), flush=True)
    assert info["version"] == "0.26.4", f"server must be 0.26.4, got {info['version']}"

    air = datasets.load_air_routes(db)
    airports = air.airports_source
    route, contains = air.route_edges_source, air.contains_edges_source
    airport_rows = db.sql(
        f"SELECT code, desc, city, country, continent, region "
        f"FROM {airports}.public.{airports}"
    ).to_pylist()
    n_route = db.sql(f"SELECT COUNT(*) n FROM {route}.public.{route}").to_pylist()[0]["n"]
    n_contains = db.sql(f"SELECT COUNT(*) n FROM {contains}.public.{contains}").to_pylist()[0]["n"]
    print(f"airports: {len(airport_rows)}  route edges: {n_route}  "
          f"contains edges: {n_contains}", flush=True)

    # Commit the airports' labels (code, continent, country, region, city) so
    # chapters recompute display-only derived numbers (homophily, cohort sizes)
    # from the cache without ever re-embedding. The embedded text is not committed.
    pq.write_table(pa.table({
        "code": [r["code"] for r in airport_rows],
        "city": [r["city"] for r in airport_rows],
        "country": [r["country"] for r in airport_rows],
        "continent": [r["continent"] for r in airport_rows],
        "region": [r["region"] for r in airport_rows],
    }), ARTIFACTS / "airports.parquet")

    metrics: dict[str, dict[str, float]] = {}

    # ---- Tier 01: embeddings + neighbor graph + homophily on continent ----
    print("\n[tier01] embed + neighbor graph", flush=True)
    emb = db.generate_embeddings(source=airports, model=EMBED_MODEL,
                                 columns=["desc", "city", "country", "region"], key="code")
    ng = db.build_neighbor_graph(airports, k=NEIGHBOR_K, exact=True)
    continent = {r["code"]: r["continent"] for r in airport_rows}
    route_rows = db.sql(f"SELECT src, dst, dist FROM {route}.public.{route}").to_pylist()
    contains_rows = db.sql(f"SELECT src, dst FROM {contains}.public.{contains}").to_pylist()

    # `route` connects airport↔airport: homophily is whether both endpoints share a
    # continent (intercontinental flights are the inhomogeneous minority).
    def route_homophily_fn(rows):
        ok = [(r["src"], r["dst"]) for r in rows
              if continent.get(r["src"]) and continent.get(r["dst"])]
        return sum(continent[s] == continent[d] for s, d in ok) / len(ok)

    # `contains` is a two-level hierarchy: every edge's child is an airport; its
    # parent is EITHER a continent code OR a country code. The source carries NO
    # continent→country edge, so a country's continent is not declared by the graph
    # (and is genuinely ambiguous for transcontinental countries — Russia, Turkey,
    # Egypt — so guessing one is a measurement artifact). The honest, non-circular
    # continent-homophily of the declared hierarchy is therefore measured over the
    # edges whose parent IS a continent code: the fraction whose child airport's
    # continent equals that parent. It is ~0.99 — the containment hierarchy is
    # (near-)perfectly continent-consistent, the few sub-1.0 cases being airports the
    # source lists under two continents. That near-1.0 against the route graph's
    # noisier ~0.83 IS the teaching contrast: a declared hierarchy carries clean
    # structure the route topology only approximates.
    continent_codes = {c for c in continent.values() if c}

    def contains_homophily_fn(rows):
        ok = [(r["src"], continent.get(r["dst"])) for r in rows
              if r["src"] in continent_codes and continent.get(r["dst"])]
        return sum(p == c for p, c in ok) / len(ok)

    route_homophily = route_homophily_fn(route_rows)
    contains_homophily = contains_homophily_fn(contains_rows)
    print(f"  homophily (continent) — route: {route_homophily:.3f}  "
          f"contains: {contains_homophily:.3f}", flush=True)
    metrics["tier01.route_homophily"] = {"value": round(route_homophily, 3), "tol": 0.02}
    metrics["tier01.contains_homophily"] = {"value": round(contains_homophily, 3), "tol": 0.02}

    golden_table, golden_url = build_continent_golden(airport_rows, query_n=200)
    db.add_source("air_continent_golden", url=golden_url, format="parquet")
    golden_rows = golden_table.to_pylist()
    query_ids = sorted({g["query_id"] for g in golden_rows})

    r1 = recall_at_k(db, emb_table=emb, golden_rows=golden_rows,
                     query_ids=query_ids, k=RECALL_K)
    print(f"  tier01 recall@{RECALL_K} (raw embeddings): {r1:.3f}", flush=True)
    metrics["tier01.recall_at_10"] = {"value": round(r1, 3), "tol": 0.03}

    _dump_emb(db, emb, ARTIFACTS / "embeddings.parquet")
    _dump_table(db, ng, ARTIFACTS / "neighbor_graph.parquet")
    pq.write_table(pa.table({"src": [r["src"] for r in route_rows],
                             "dst": [r["dst"] for r in route_rows],
                             "dist": [float(r["dist"]) for r in route_rows]}),
                   ARTIFACTS / "route_edges.parquet")
    pq.write_table(pa.table({"src": [r["src"] for r in contains_rows],
                             "dst": [r["dst"] for r in contains_rows]}),
                   ARTIFACTS / "contains_edges.parquet")

    # ---- Tier 02: propagate over the declared route graph (low-pass filter) ----
    print("\n[tier02] propagate over route (degree_normalized, undirected)", flush=True)
    prop = db.propagate_embeddings(source=airports, embedding_table=emb, edge_source=route,
                                   edge_src_column="src", edge_dst_column="dst",
                                   direction="undirected", hops=PROP_HOPS,
                                   weighting="degree_normalized")
    r2 = recall_at_k(db, emb_table=prop, golden_rows=golden_rows,
                     query_ids=query_ids, k=RECALL_K)
    print(f"  tier02 recall@{RECALL_K} (propagated): {r2:.3f}  (Δ {r2 - r1:+.3f})", flush=True)
    metrics["tier02.recall_at_10"] = {"value": round(r2, 3), "tol": 0.03}
    metrics["tier02.recall_delta"] = {"value": round(r2 - r1, 3), "tol": 0.03}
    _dump_emb(db, prop, ARTIFACTS / "propagated.parquet")

    # ---- The tenancy showcase: the two genuine isolation layers + the caveat ----
    print("\n[tenancy] catalog-listing + discriminator-column isolation", flush=True)
    record = tenancy_showcase(target, airport_rows)
    # Catalog-listing isolation: tenant A's source listing excludes B's source.
    metrics["tenancy.listing_leak"] = {"value": float(record["listing_leak"]), "tol": 0.0}
    # Row-level discriminator-column isolation: a shared tenant_id-tagged source
    # returns disjoint rows under A vs B (zero cross-tenant rows surfaced).
    metrics["tenancy.discriminator_leak"] = {
        "value": float(record["discriminator_leak"]), "tol": 0.0}
    # The honest caveat (a POSITIVE assertion): a discriminator-less source is
    # globally readable — tenant A sees ALL of B's rows when it names the source.
    metrics["tenancy.global_source_visible"] = {
        "value": float(record["global_source_visible"]), "tol": 0.0}

    (ARTIFACTS / "golden_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    _write_checksums()
    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


def tenancy_showcase(target: str, airport_rows: list[dict]) -> dict:
    """Record the engine's two genuine tenant-isolation layers + the honest caveat.

    The engine isolates by tenant at two layers; "a separate source per tenant" is
    NOT data isolation. ``set_tenant`` binds the scope to the connection in place;
    the bound scope drives both layers:

    1. **Catalog-listing isolation** — ``list_sources`` filters the registry to
       ``tenant_id = $cur OR IS NULL``, so tenant A's listing excludes a source
       registered under tenant B.
    2. **Row-level discriminator-column isolation** — the analyzer injects
       ``tenant_id = $cur OR IS NULL`` onto a ``TableScan`` *only* when the queried
       table's schema carries a ``tenant_id`` column, so the SAME tagged source
       returns disjoint rows under A vs B.

    And the caveat, stated honestly as a positive observation: a source with NO
    discriminator column is **globally readable** by any tenant that names it — the
    engine does not authenticate (access-gating lives above it). Returns the
    recorded counts (each cross-tenant leak is a hard zero; the global-source
    visibility is the full foreign row count, by design).
    """
    na = sorted(r["code"] for r in airport_rows if r["continent"] == "NA")
    eu = sorted(r["code"] for r in airport_rows if r["continent"] == "EU")
    work = ARTIFACTS / "_tenancy_work"
    work.mkdir(parents=True, exist_ok=True)

    db = jammi_ai.connect(target)

    def write_src(name: str, table: pa.Table) -> None:
        path = work / f"{name}.parquet"
        pq.write_table(table, path)
        db.add_source(name, url=str(path), format="parquet")

    # --- Layer 1: catalog-listing isolation -------------------------------------
    db.set_tenant(TENANT_A)
    write_src("air_na", pa.table({"code": na}))
    db.set_tenant(TENANT_B)
    write_src("air_eu", pa.table({"code": eu}))
    db.set_tenant(TENANT_A)
    a_listed = {s["source_id"] for s in db.list_sources()}
    listing_leak = len({"air_eu"} & a_listed)
    print(f"  [1] tenant A lists {sorted(a_listed)}; B's 'air_eu' leaked: "
          f"{listing_leak}", flush=True)

    # --- Layer 2: row-level discriminator-column isolation ----------------------
    db.set_tenant("")  # register the shared source as a global (tenant_id IS NULL)
    write_src("air_tagged", pa.table({
        "code": na + eu,
        "tenant_id": [TENANT_A] * len(na) + [TENANT_B] * len(eu),
    }))
    db.set_tenant(TENANT_A)
    seen_a = {r["code"] for r in db.sql(
        "SELECT code FROM air_tagged.public.air_tagged").to_pylist()}
    discriminator_leak = len(seen_a & set(eu))  # A must surface no B-tagged row
    print(f"  [2] tenant A reads tagged source: {len(seen_a)} rows; "
          f"B-tagged rows leaked: {discriminator_leak}", flush=True)

    # --- The honest caveat: a discriminator-less source is globally readable -----
    db.set_tenant(TENANT_B)
    write_src("air_eu_nodisc", pa.table({"code": eu}))
    db.set_tenant(TENANT_A)
    seen_global = {r["code"] for r in db.sql(
        "SELECT code FROM air_eu_nodisc.public.air_eu_nodisc").to_pylist()}
    global_source_visible = len(seen_global & set(eu))  # ALL of B's rows, by design
    print(f"  [caveat] tenant A reads B's discriminator-less source: "
          f"{global_source_visible} of {len(eu)} EU rows visible (by design)",
          flush=True)

    record = {
        "tenant_a": TENANT_A, "tenant_b": TENANT_B,
        "tenant_a_continent": "NA", "tenant_b_continent": "EU",
        "tenant_a_airports": len(na), "tenant_b_airports": len(eu),
        "listing_leak": listing_leak,
        "discriminator_leak": discriminator_leak,
        "discriminator_rows_seen": len(seen_a),
        "global_source_visible": global_source_visible,
        "note": "The engine's two genuine tenant-isolation layers + the honest "
                "caveat. (1) Catalog-listing isolation: tenant A's list_sources "
                "excludes B's registered source (hard zero). (2) Row-level "
                "discriminator-column isolation: the analyzer injects "
                "tenant_id = $cur OR IS NULL onto a TableScan only when the table "
                "carries a tenant_id column, so the SAME tagged source returns "
                "disjoint rows under A vs B (zero B-tagged rows surfaced to A). "
                "Caveat: a discriminator-less source is GLOBALLY READABLE — tenant "
                "A sees all of B's rows when it names the source. The engine does "
                "not authenticate; access-gating lives above it. "
                "set_tenant binds the scope in place and does NOT "
                "make a discriminator-less source's data invisible.",
    }
    (ARTIFACTS / "tenancy.json").write_text(json.dumps(record, indent=2))
    return record


def _write_checksums() -> None:
    sums = {p.name: _checksum(p) for p in sorted(ARTIFACTS.glob("*"))
            if p.is_file() and p.name != "checksums.json"}
    (ARTIFACTS / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="grpc://127.0.0.1:50051",
                    help="connect() target — grpc://host:port for the GPU server.")
    args = ap.parse_args()
    db = jammi_ai.connect(args.target)
    emit(db, args.target)


if __name__ == "__main__":
    main()
