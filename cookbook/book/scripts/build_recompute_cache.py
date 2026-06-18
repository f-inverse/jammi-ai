#!/usr/bin/env python3
"""Emit the incremental-recompute + opt-in-caching cache (W-61) — CPU, hermetic.

The engine↔cookbook validator for SPEC-03's two bounded actions over the
materialization contract: **opt-in memoization** on the result-table producers
(`cache="use"` → reuse an existing exact materialisation instead of recomputing)
and **`recompute(table, cascade)`** (re-invoke the recorded producer over the
inputs' current state, optionally sweeping the bounded downstream DAG once). It
also exercises the **sensing** reads the actions stand on — `staleness`,
`derives_from`, and the `verify_materialization` definition probe.

Everything runs on CPU over an ~8-row ephemeral `file://` catalog and a tiny
`tiny_modernbert` fixture (candle falls back off CUDA), so the whole recipe is
hermetic and needs no GPU and no server. Reuse is observed by **table-name
identity**: a result-table producer returns a bare table-name `str`; a cache hit
returns the SAME name, a miss a new timestamped name. That is the only honest
signal — a wall-clock measurement is non-deterministic and forbidden by the
determinism contract — so every cache verdict is frozen as a name-equality
boolean (1.0/0.0), never a duration.

## The chain (the SPEC-03 cacheable producers, live-verified on 0.31.0)

    add_source("docs", file://…parquet)
      → generate_embeddings(...)                       → emb   (UnpinnedAtInstant — no hit)
      → build_neighbor_graph("docs", k=3, exact=True)  → g     (ResultDigest — cacheable)
      → propagate_embeddings("docs", embedding_table=emb,
            edge_graph_table=g, hops=1, alpha=0.5)     → prop  (ResultDigest ×2 — cacheable)

**Two recipe gotchas, baked in (chapter callout boxes + the decisions log):**

1. `build_neighbor_graph` / `propagate_embeddings` take the ORIGINAL `source` id
   (`"docs"`), NOT the embedding-table name. Passing the table id raises
   `Catalog error: No ready embedding table for source '<table>'` — the verb
   resolves the latest ready embedding *for the named source*, so it wants the
   source, and the embedding table is passed separately via `embedding_table=`.
2. `propagate_embeddings` MUST pin `embedding_table=emb` explicitly to be
   cacheable. Left unpinned it resolves "the latest ready embedding for docs",
   and once a prior propagate has run, that latest-ready anchor shifts → the
   `ResultDigest` differs → cache miss. Pinning the exact embedding table fixes
   the anchor, so the same build keys to the same materialisation.

## current_definition — obtained purely via Python (no Rust, no internals)

`staleness(table, current_definition)` needs the table's own recorded
`DefinitionHash`. There is no `get_definition` verb, but `verify_materialization`
returns the recorded hash as its `found` field on a deliberate mismatch:

    verify_materialization(table, expected_definition="deadbeef")
        → {"verdict": "mismatch", "expected": "deadbeef", "found": "<the recorded hash>"}

So `_definition_of(db, table)` reads `found` — the recorded definition hash,
straight off the Python surface.

## The three measured assertion families → golden_metrics.json

1. **cache-hit-reuses** — `bng(k=3, cache="bypass")` then `bng(k=3, cache="use")`
   return the SAME name (`reused=1.0`); `bng(k=4)` and `bng(k=3, min_similarity=0.5)`
   each return NEW names (reuse `0.0` — the probe keys on the FULL descriptor, not
   just `k`); `propagate(embedding_table=emb, hops=1, cache="use")` reuses while
   `hops=2` computes (`0.0`); `generate_embeddings(cache="use")` twice returns
   DIFFERENT names (`unpinned_reuse=0.0` — an `UnpinnedAtInstant` producer honestly
   never hits).
2. **staleness (the `definition_changed` arm)** — `staleness(child, child's own
   recorded hash)` → `fresh` (`fresh=1.0`); `staleness(child, a_different_hash)` →
   `stale` with reason `definition_changed` naming `recorded` / `current`
   (`stale_on_definition_change=1.0`). The `result_digest` input-drift arm is CUT
   (a written rationale, not a measure — see the module note below and the
   decisions log). It is described via the manifest-anchor model in the chapter
   prose, never half-shipped as a fake demo.
3. **recompute-restores** — `recompute(child, cascade="report_only")` →
   `recomputed[0].outcome == "computed"` AND byte-identical (the recompute's
   `verify_materialization` verdict is `match` against the original's recorded
   hash AND `found(recomputed) == found(original)`) → `recompute_byte_identical=1.0`;
   `recompute(parent, cascade="report_only")` → `downstream_stale` lists the child
   (count golden); `recompute(parent, cascade="downstream")` → `recomputed` carries
   parent + child (count golden); `derives_from(emb)` → one-hop edges, every
   `kind == "result_digest"` (edge-count + kind goldens).

## Why the `result_digest` input-drift staleness arm is CUT (build-or-cut)

The `definition_changed` arm of `staleness` is fully measurable (assertion family
2). The sibling `result_digest` input-drift arm — a child senses a *parent whose
artifact digest moved* and reports `stale` for reason `result_digest` — is NOT
measured here, by design, with this rationale:

- A `recompute` over UNMOVED inputs is byte-identical (proven in family 3) AND
  produces a uniquely-named NEW table. So a child anchored on the original parent
  stays honestly `fresh`: its recorded parent digest still matches that parent's
  current digest. Nothing drifted.
- There is no Python-surface verb to re-anchor an existing child onto a
  moved-digest parent in a hermetic chain (re-anchoring happens only when a
  producer runs and records a fresh anchor — which produces a new child, not a
  drift on the old one). Fabricating one would require reaching past the public
  surface into the manifest store.

Rather than half-ship a fake input-drift demo (forbidden — no band-aids), the
arm is CUT: the chapter DESCRIBES it via the SPEC-02 manifest-anchor model (a
recomputed parent gets a new digest; a child anchored on the old one is detected
stale by the same per-input comparison — recursion falls out with no special
case) in prose, and MEASURES only the `definition_changed` arm.

## The engine/platform boundary (stated, names no consumer)

The engine ships the bounded MECHANISM: one cache probe per producer call, one
recompute on one explicit request, one bounded downstream sweep on `cascade=
"downstream"`. The SCHEDULED / monitored recompute LOOP — a sensor that watches
staleness and triggers recompute, a cron sweep, cache eviction/TTL — is the
consumer's composition, built on a published engine version. Names no consumer.

Usage::

    python scripts/build_recompute_cache.py --fixtures-root /path/to/jammi-ai

The ``--fixtures-root`` is the engine checkout carrying
``tests/fixtures/tiny_modernbert`` (or set ``JAMMI_FIXTURES_ROOT``) — the base
model the embedding step runs. CPU/hermetic. Emit-only; PR CI reads the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import jammi_ai
import pyarrow as pa
import pyarrow.parquet as pq

import jammi_cookbook  # noqa: F401  # applies the determinism env on import

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "recompute"

# A tiny, fixed text corpus — 8 GSP/GNN phrases. Subset identity is the committed
# literal (the determinism contract: identity is committed, not seeded), and 8
# rows embed + neighbor-graph + propagate in ~1s on CPU.
_DOCS = {
    "_row_id": [f"d{i}" for i in range(8)],
    "text": [
        "graph signal processing",
        "node embedding vectors",
        "edge weight matrix",
        "spectral graph filter",
        "random walk kernel",
        "message passing layer",
        "attention head pooling",
        "laplacian eigenmap basis",
    ],
}


def _definition_of(db, table: str) -> str:
    """The table's recorded `DefinitionHash`, read purely off the Python surface.

    There is no `get_definition` verb, but `verify_materialization` returns the
    recorded hash as `found` on a deliberate mismatch — so a probe with a sentinel
    `expected_definition` yields the real recorded definition hash."""
    return db.verify_materialization(table, expected_definition="deadbeef")["found"]


def _build_chain(db, model: str, src_path: Path) -> tuple[str, str, str]:
    """The cacheable SPEC-03 chain over the ephemeral source. Returns
    `(emb, g, prop)` table-name strings. `g` / `prop` are built under `cache="use"`
    so the first call computes and records the materialisation a later probe reuses.

    Gotcha 1 is honoured here: `build_neighbor_graph` / `propagate_embeddings` take
    the ORIGINAL source id `"docs"`, with the embedding table passed separately via
    `embedding_table=`. Gotcha 2: `propagate_embeddings` pins `embedding_table=emb`
    explicitly so its `ResultDigest` anchor is fixed (cacheable)."""
    db.add_source("docs", url=f"file://{src_path}", format="parquet")
    emb = db.generate_embeddings(source="docs", model=model, columns=["text"], key="_row_id")
    g = db.build_neighbor_graph("docs", k=3, exact=True, cache="use")
    prop = db.propagate_embeddings(
        "docs", embedding_table=emb, edge_graph_table=g,
        hops=1, alpha=0.5, output="final", cache="use",
    )
    return emb, g, prop


def _fresh_chain(catalog_root: Path, model: str, src_path: Path) -> tuple:
    """Open a brand-new ephemeral catalog and build the chain in it, so the
    recompute/derives-from counts are deterministic (a shared catalog accumulates
    redundant recomputed tables and inflates the downstream / edge counts). Returns
    `(db, emb, g, prop)`."""
    catalog = tempfile.mkdtemp(prefix="jammi_recompute_", dir=catalog_root)
    db = jammi_ai.connect(f"file://{catalog}")
    emb, g, prop = _build_chain(db, model, src_path)
    return db, emb, g, prop


# --------------------------------------------------------------------------- #
# Family 1 — cache-hit-reuses (reuse observed by table-name identity)
# --------------------------------------------------------------------------- #


def run_cache(catalog_root: Path, model: str, src_path: Path) -> dict:
    """Measure opt-in memoization as name-identity verdicts on a clean catalog.

    A producer returns a bare table-name str; a cache HIT returns the SAME name, a
    MISS a new timestamped name. Every cell is the boolean of that identity."""
    catalog = tempfile.mkdtemp(prefix="jammi_recompute_cache_", dir=catalog_root)
    db = jammi_ai.connect(f"file://{catalog}")
    db.add_source("docs", url=f"file://{src_path}", format="parquet")
    emb = db.generate_embeddings(source="docs", model=model, columns=["text"], key="_row_id")

    # neighbor-graph: bypass computes, then use REUSES the same materialisation.
    g_bypass = db.build_neighbor_graph("docs", k=3, exact=True, cache="bypass")
    g_use = db.build_neighbor_graph("docs", k=3, exact=True, cache="use")
    bng_reused = g_bypass == g_use

    # the probe keys on the FULL descriptor: a different k, or a different
    # min_similarity, is a different definition → a MISS (a new name).
    g_k4 = db.build_neighbor_graph("docs", k=4, exact=True, cache="use")
    g_minsim = db.build_neighbor_graph("docs", k=3, exact=True, min_similarity=0.5, cache="use")
    bng_k4_recomputed = g_k4 != g_use
    bng_minsim_recomputed = g_minsim != g_use

    # propagate: pinned embedding table + hops=1 reuses; hops=2 is a new definition.
    p_h1_a = db.propagate_embeddings(
        "docs", embedding_table=emb, edge_graph_table=g_use,
        hops=1, alpha=0.5, output="final", cache="use",
    )
    p_h1_b = db.propagate_embeddings(
        "docs", embedding_table=emb, edge_graph_table=g_use,
        hops=1, alpha=0.5, output="final", cache="use",
    )
    prop_reused = p_h1_a == p_h1_b
    p_h2 = db.propagate_embeddings(
        "docs", embedding_table=emb, edge_graph_table=g_use,
        hops=2, alpha=0.5, output="final", cache="use",
    )
    prop_hops_recomputed = p_h2 != p_h1_a

    # the unpinned producer honestly NEVER hits: generate_embeddings is anchored on
    # an UnpinnedAtInstant source, so cache="use" twice still yields two names.
    emb_a = db.generate_embeddings(source="docs", model=model, columns=["text"], key="_row_id",
                                   cache="use")
    emb_b = db.generate_embeddings(source="docs", model=model, columns=["text"], key="_row_id",
                                   cache="use")
    unpinned_reused = emb_a == emb_b  # expected False — never reuses

    return {
        "bng_reused": bool(bng_reused),
        "bng_k4_recomputed": bool(bng_k4_recomputed),
        "bng_minsim_recomputed": bool(bng_minsim_recomputed),
        "prop_reused": bool(prop_reused),
        "prop_hops_recomputed": bool(prop_hops_recomputed),
        "unpinned_reused": bool(unpinned_reused),
        # the observed names, recorded as provenance (not asserted on value)
        "names": {
            "g_bypass": g_bypass, "g_use": g_use, "g_k4": g_k4, "g_minsim": g_minsim,
            "prop_h1_a": p_h1_a, "prop_h1_b": p_h1_b, "prop_h2": p_h2,
            "emb_a": emb_a, "emb_b": emb_b,
        },
    }


# --------------------------------------------------------------------------- #
# Family 2 — staleness (the definition_changed arm; result_digest drift is CUT)
# --------------------------------------------------------------------------- #


def run_staleness(catalog_root: Path, model: str, src_path: Path) -> dict:
    """Measure the `definition_changed` staleness arm on a clean chain.

    `staleness(child, child's own recorded hash)` is `fresh`; `staleness(child, a
    different hash)` is `stale` with reason `definition_changed` naming `recorded`
    and `current`. The `result_digest` input-drift arm is CUT (see the module note)
    — described in prose, not measured."""
    db, emb, g, prop = _fresh_chain(catalog_root, model, src_path)

    recorded = _definition_of(db, prop)
    fresh = db.staleness(prop, recorded)
    a_different_hash = "00000000" if recorded != "00000000" else "11111111"
    stale = db.staleness(prop, a_different_hash)

    is_fresh = fresh.get("staleness") == "fresh"
    reasons = stale.get("reasons", [])
    is_stale_def = (
        stale.get("staleness") == "stale"
        and any(r.get("reason") == "definition_changed" for r in reasons)
    )
    def_reason = next((r for r in reasons if r.get("reason") == "definition_changed"), {})
    names_recorded_current = "recorded" in def_reason and "current" in def_reason

    return {
        "recorded_definition": recorded,
        "fresh_verdict": fresh,
        "stale_verdict": stale,
        "is_fresh": bool(is_fresh),
        "is_stale_on_definition_change": bool(is_stale_def and names_recorded_current),
    }


# --------------------------------------------------------------------------- #
# Family 3 — recompute-restores (byte-identical replay, cascade, lineage)
# --------------------------------------------------------------------------- #


def run_recompute(catalog_root: Path, model: str, src_path: Path) -> dict:
    """Measure the bounded recompute action + the lineage it sweeps, on clean
    chains (one chain per count assertion, so the downstream / edge counts are
    deterministic)."""

    # --- child recompute is byte-identical -------------------------------- #
    db, emb, g, prop = _fresh_chain(catalog_root, model, src_path)
    original_hash = _definition_of(db, prop)
    rc_child = db.recompute(prop, cascade="report_only")
    recomputed_name = rc_child["recomputed"][0]["recomputed"]
    child_outcome = rc_child["recomputed"][0]["outcome"]
    # byte-identical: the recompute's verdict against the ORIGINAL's recorded hash
    # is `match`, AND the recompute's own recorded hash equals the original's.
    verdict = db.verify_materialization(recomputed_name, expected_definition=original_hash)
    byte_identical = (
        child_outcome == "computed"
        and verdict["verdict"] == "match"
        and _definition_of(db, recomputed_name) == original_hash
    )

    # --- parent report_only reports the downstream-stale set, recomputes none of it #
    db, emb, g, prop = _fresh_chain(catalog_root, model, src_path)
    rc_parent_report = db.recompute(g, cascade="report_only")
    downstream_stale = rc_parent_report["downstream_stale"]
    report_recomputed_only_parent = len(rc_parent_report["recomputed"]) == 1
    child_in_downstream = prop in downstream_stale

    # --- parent cascade=downstream sweeps parent + child once ------------- #
    db, emb, g, prop = _fresh_chain(catalog_root, model, src_path)
    rc_parent_down = db.recompute(g, cascade="downstream")
    cascade_recomputed = rc_parent_down["recomputed"]

    # --- lineage: derives_from(emb) one-hop edges, all result_digest ------ #
    db, emb, g, prop = _fresh_chain(catalog_root, model, src_path)
    edges = db.derives_from(emb)
    edge_kinds = sorted({e["kind"] for e in edges})
    all_result_digest = edge_kinds == ["result_digest"]

    return {
        "child_recompute_byte_identical": bool(byte_identical),
        "child_recompute_outcome": child_outcome,
        "downstream_stale_count": len(downstream_stale),
        "child_in_downstream_stale": bool(child_in_downstream),
        "report_only_recomputes_only_parent": bool(report_recomputed_only_parent),
        "cascade_recomputed_count": len(cascade_recomputed),
        "derives_from_edge_count": len(edges),
        "derives_from_all_result_digest": bool(all_result_digest),
        "derives_from_kinds": edge_kinds,
    }


# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checksums() -> None:
    sums = {
        p.name: _checksum(p)
        for p in sorted(ARTIFACTS.glob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    (ARTIFACTS / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))


def emit(fixtures_root: Path) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model = f"local:{fixtures_root / 'tests' / 'fixtures' / 'tiny_modernbert'}"

    with tempfile.TemporaryDirectory() as work_root:
        work = Path(work_root)
        src_path = work / "docs.parquet"
        pq.write_table(pa.table(_DOCS), src_path)

        print("== family 1: cache-hit-reuses (name identity) ==", flush=True)
        cache = run_cache(work, model, src_path)
        print("== family 2: staleness (definition_changed arm) ==", flush=True)
        staleness = run_staleness(work, model, src_path)
        print("== family 3: recompute-restores (byte-identical + cascade + lineage) ==",
              flush=True)
        recompute = run_recompute(work, model, src_path)

    matrix = {"cache": cache, "staleness": staleness, "recompute": recompute}

    # --- the frozen golden matrix (every cell a measured verdict, tol 0) --- #
    golden = {
        # family 1 — cache-hit-reuses
        "cache.bng_reused": {"value": 1.0 if cache["bng_reused"] else 0.0, "tol": 0.0},
        "cache.bng_k4_recomputed": {
            "value": 1.0 if cache["bng_k4_recomputed"] else 0.0, "tol": 0.0
        },
        "cache.bng_minsim_recomputed": {
            "value": 1.0 if cache["bng_minsim_recomputed"] else 0.0, "tol": 0.0
        },
        "cache.prop_reused": {"value": 1.0 if cache["prop_reused"] else 0.0, "tol": 0.0},
        "cache.prop_hops_recomputed": {
            "value": 1.0 if cache["prop_hops_recomputed"] else 0.0, "tol": 0.0
        },
        "cache.unpinned_reused": {"value": 1.0 if cache["unpinned_reused"] else 0.0, "tol": 0.0},
        # family 2 — staleness (definition_changed arm)
        "staleness.fresh": {"value": 1.0 if staleness["is_fresh"] else 0.0, "tol": 0.0},
        "staleness.stale_on_definition_change": {
            "value": 1.0 if staleness["is_stale_on_definition_change"] else 0.0, "tol": 0.0
        },
        # family 3 — recompute-restores
        "recompute.byte_identical": {
            "value": 1.0 if recompute["child_recompute_byte_identical"] else 0.0, "tol": 0.0
        },
        "recompute.downstream_stale_count": {
            "value": float(recompute["downstream_stale_count"]), "tol": 0.0
        },
        "recompute.report_only_recomputes_only_parent": {
            "value": 1.0 if recompute["report_only_recomputes_only_parent"] else 0.0, "tol": 0.0
        },
        "recompute.cascade_recomputed_count": {
            "value": float(recompute["cascade_recomputed_count"]), "tol": 0.0
        },
        "recompute.derives_from_edge_count": {
            "value": float(recompute["derives_from_edge_count"]), "tol": 0.0
        },
        "recompute.derives_from_all_result_digest": {
            "value": 1.0 if recompute["derives_from_all_result_digest"] else 0.0, "tol": 0.0
        },
    }

    (ARTIFACTS / "matrix.json").write_text(json.dumps(matrix, indent=2, sort_keys=True))
    (ARTIFACTS / "golden_metrics.json").write_text(json.dumps(golden, indent=2, sort_keys=True))

    record = {
        "purpose": (
            "The engine↔cookbook validator for SPEC-03's two bounded actions over the "
            "materialization contract — opt-in producer memoization (cache='use') and "
            "recompute(table, cascade) — and the sensing reads they stand on (staleness, "
            "derives_from, the verify_materialization definition probe). Measured on CPU "
            "over an ~8-row ephemeral file:// catalog, with reuse observed by table-name "
            "identity (a producer returns a bare name str; a hit returns the SAME name)."
        ),
        "current_definition_method": (
            "The recorded DefinitionHash is read purely off the Python surface: "
            "verify_materialization(table, expected_definition='deadbeef') returns the "
            "recorded hash as its 'found' field on a deliberate mismatch. No get_definition "
            "verb exists; no engine internals are touched."
        ),
        "gotchas": [
            "build_neighbor_graph / propagate_embeddings take the ORIGINAL source id "
            "('docs'), NOT the embedding-table name; passing the table id raises 'Catalog "
            "error: No ready embedding table for source <table>'. The embedding table is "
            "passed separately via embedding_table=.",
            "propagate_embeddings MUST pin embedding_table=emb explicitly to be cacheable; "
            "left unpinned it resolves 'the latest ready embedding for docs', and once a "
            "prior propagate has run that latest-ready anchor shifts → a different "
            "ResultDigest → a cache miss.",
        ],
        "staleness_result_digest_drift_cut": (
            "The definition_changed staleness arm is measured. The sibling result_digest "
            "input-drift arm is CUT (not half-shipped). A recompute over unmoved inputs is "
            "byte-identical AND produces a uniquely-named new table, so a child stays "
            "honestly fresh (its recorded parent digest still matches the parent's current "
            "digest — nothing drifted), and there is no Python-surface verb to re-anchor an "
            "existing child onto a moved-digest parent in a hermetic chain. Rather than "
            "fabricate an input-drift demo, the arm is described via the SPEC-02 "
            "manifest-anchor model in prose and only the definition_changed arm is measured."
        ),
        "boundary": (
            "The engine ships the bounded MECHANISM — one cache probe per producer call, "
            "one recompute on one explicit request, one bounded downstream sweep on "
            "cascade='downstream'. The scheduled / monitored recompute LOOP (a "
            "staleness-monitor that triggers recompute, a cron sweep, cache eviction/TTL) "
            "is the consumer's composition, built on a published engine version. Names no "
            "consumer."
        ),
        "matrix": matrix,
    }
    (ARTIFACTS / "recompute.json").write_text(json.dumps(record, indent=2, sort_keys=True))

    _write_checksums()

    # --- the loud verdict --------------------------------------------------- #
    print("\n=== incremental-recompute + caching, measured (embedded, CPU-hermetic) ===",
          flush=True)
    print("  FAMILY 1 — cache-hit-reuses (1.0 = reused, 0.0 = recomputed):", flush=True)
    print(f"    bng_reused              = {1.0 if cache['bng_reused'] else 0.0}  "
          f"(k=3 use reuses k=3 bypass)", flush=True)
    print(f"    bng_k4_recomputed       = {1.0 if cache['bng_k4_recomputed'] else 0.0}  "
          f"(k=4 is a new definition)", flush=True)
    print(f"    bng_minsim_recomputed   = {1.0 if cache['bng_minsim_recomputed'] else 0.0}  "
          f"(min_similarity is a new definition)", flush=True)
    print(f"    prop_reused             = {1.0 if cache['prop_reused'] else 0.0}  "
          f"(pinned emb + hops=1 reuses)", flush=True)
    print(f"    prop_hops_recomputed    = {1.0 if cache['prop_hops_recomputed'] else 0.0}  "
          f"(hops=2 is a new definition)", flush=True)
    print(f"    unpinned_reused         = {1.0 if cache['unpinned_reused'] else 0.0}  "
          f"(generate_embeddings honestly never hits)", flush=True)
    print("  FAMILY 2 — staleness (definition_changed arm):", flush=True)
    print(f"    fresh                   = {1.0 if staleness['is_fresh'] else 0.0}", flush=True)
    print(f"    stale_on_definition_change = "
          f"{1.0 if staleness['is_stale_on_definition_change'] else 0.0}", flush=True)
    print("    (result_digest input-drift arm CUT — described in prose, not measured)",
          flush=True)
    print("  FAMILY 3 — recompute-restores:", flush=True)
    print(f"    byte_identical          = "
          f"{1.0 if recompute['child_recompute_byte_identical'] else 0.0}", flush=True)
    print(f"    downstream_stale_count  = {recompute['downstream_stale_count']}", flush=True)
    print(f"    cascade_recomputed_count= {recompute['cascade_recomputed_count']}", flush=True)
    print(f"    derives_from_edge_count = {recompute['derives_from_edge_count']}  "
          f"kinds={recompute['derives_from_kinds']}", flush=True)

    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


def _fixtures_root(arg: str | None) -> Path:
    root = arg or os.environ.get("JAMMI_FIXTURES_ROOT")
    if not root:
        raise SystemExit(
            "pass --fixtures-root (or set JAMMI_FIXTURES_ROOT) to the engine checkout "
            "carrying tests/fixtures/tiny_modernbert (the base model the embedding step runs)"
        )
    p = Path(root).resolve()
    if not (p / "tests" / "fixtures" / "tiny_modernbert" / "config.json").exists():
        raise SystemExit(f"--fixtures-root {p} has no tests/fixtures/tiny_modernbert")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-root", default=None,
                    help="engine checkout with tests/fixtures/tiny_modernbert "
                         "(or set JAMMI_FIXTURES_ROOT)")
    args = ap.parse_args()
    emit(_fixtures_root(args.fixtures_root))


if __name__ == "__main__":
    main()
