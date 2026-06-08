# KV-air — the Air Routes vertical (tiers 01–02 on-ramp)

**Status:** spec — draft (hand-off). Build after KV-arxiv is CI-green (so the shared lib + contracts + patterns are locked). One PR.
**Why Air Routes, and only 01–02:** it is Neptune's own teaching dataset — small (~3.4k nodes / ~50k edges), legible, hub-and-spoke, with two clean edge types (`route`) and a built-in hierarchy (`contains`). It maps directly to the monograph's Part I/II and lets the book open with the "here is Neptune's organization, re-expressed as runnable Jammi computation" move. Its text is too thin and its label (continent ≈ solved by lat/lon) too toy for a credible tier-03/04 — so this vertical **stops at 02**; tiers 03–04 live on ogbn-arxiv (KV-arxiv). Do not force a tier-04 here.

This is the **Neptune-contrast on-ramp**: each recipe should note "Neptune demos this with `CALL neptune.algo.*` on Air Routes; here it is as a Jammi recipe."

---

## Pipeline (serial; reads the locked lib/contracts; light execution — runs fast on CPU)

Dataset: `load_air_routes()` (K2) — airport node source + `route` and `contains` edge sources.

### Tier 01 — Construct
- **Substrate:** embed airport rows (numeric/categorical features + the thin `desc`/`city` text) → `air.embeddings`; `build_neighbor_graph(air.embeddings, k=…, exact=True)` → `air.neighbor_graph`. Register `route` and `contains` as declared edge sources.
- **Theory:** topology from data (monograph Part I); the property graph + hierarchy.
- **Bridge / Neptune-contrast:** Neptune *loads* a given graph; Jammi *constructs* the similarity graph and *registers* the declared `route`/`contains` graphs side by side — the two-graphs contrast in miniature.
- **Rail (tenancy):** load a second region (e.g. a different continent) under a **second tenant**; assert the two-tenant test — a query under tenant A never sees tenant B's airports/edges. (Air Routes is the clean place to make the tenancy rail vivid.)
- **Measure:** degree distribution; homophily of `route` vs `contains` on the continent label.

### Tier 02 — Analyze
- **Substrate:** graph algorithms via `sql()` / `assemble_context` edge-pooling + `db.propagate_embeddings(source=airports, edge_source=route, direction="undirected", hops=2, weighting="degree_normalized")` over the `route` graph; `search` for "airports like this one." Show pathfinding/centrality/community framing (the Neptune Analytics set) expressed as Jammi queries/aggregates.
- **Theory:** graph signal processing — low-pass filtering / smoothing over the route graph (monograph Part II) = SGC.
- **Bridge / Neptune-contrast:** Neptune Analytics ships `pageRank`/`degree`/`wcc`/`jaccard` as `CALL`s on Air Routes; here, propagation-as-low-pass-filter and similarity are Jammi recipes — same operations, expressed as substrate computation, with the spectral=message-passing=aggregate bridge made explicit (coordinate with K-bridge).
- **Rail (provenance + measurement):** propagated features carry their derivation; `eval_compare` propagated vs raw on a simple retrieval task (continent recovery) — a clean, honest small-data measurement, explicitly noted as a *teaching* label not a literature benchmark.

## Determinism contract

Embed K0 §3. Air Routes is small enough to run exactly and fast; `exact=True`, committed IDs (or the full small graph), pinned embed model, tolerances. Commit `artifacts/air/` (embeddings, neighbor_graph, propagated, golden metrics) for CI.

## Out of scope

No tier-03/04 here (thin text, toy label). No general traversal verb (Jammi retracted it; use bounded gather / recursive SQL framing only as a *consumer* note). Don't manufacture a conformal story on a near-deterministic continent label.

## Success criteria

Tiers 01–02 run `--execute` CI-green on the committed Air Routes subset; the two-tenant test passes (the tenancy rail's showcase); the Neptune-contrast note appears in each recipe; propagation-vs-raw measured with a real number; bridge notes present (deepened by K-bridge); no band-aids; API calls match the K1 reference.
