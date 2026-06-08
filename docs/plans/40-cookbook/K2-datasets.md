# K2 — Dataset specs (Air Routes + ogbn-arxiv)

**Status:** spec — draft (hand-off). After K1. One PR (the two loaders are trivially independent and may share it).
**Produces:** `jammi_cookbook/datasets.py` loaders + the committed subset ID lists in `data/ids/`.

---

## Why two datasets (the hand-off / each where it's strong)

- **Air Routes** (Neptune's own dataset) — airports/countries/continents nodes; `route` (airport↔airport, `dist`) + `contains` (continent→country→airport hierarchy) edges. ~3.4k airports / ~50k edges. Numeric/categorical features (lat, lon, elev, runways, country, continent), **thin text** (`desc`, `city`). → the **tiers 01–02 on-ramp**: graph construction + graph algorithms + propagation, mapping to the monograph's Part I/II. It is what Neptune itself uses for queries/algorithms (and Neptune switches datasets for ML — so do we).
- **ogbn-arxiv** — ~169k CS papers, ~1.16M citation edges, 40 subject-area labels, rich title+abstract text. → the **tiers 03–04 spine**: text embeddings → fine-tune → graph-conditioned context → conformal. It is **the** dataset the graph-conformal literature (CF-GNN, NAPS) reports on, so the tier-04 theory bridge is directly comparable to published results.

## Determinism (K0 §3) — load-bearing for datasets

- **Pin by version + checksum.** Air Routes: pin the `air-routes-latest` GraphML/CSV by commit/checksum from the canonical `krlawrence/graph` `sample-data/`. ogbn-arxiv: pin the OGB release + checksum-gate the download (OGB has reissued splits before).
- **Commit the subset ID lists.** Do **not** rely on "seed=N reproduces the same nodes" across library versions — it won't. The loader selects the subset with a recorded seed **and writes the selected `_row_id` list to `data/ids/<dataset>.txt`, committed**; CI loads from that file. The seed is recorded for provenance; the committed list is the source of truth.
- **Small fixed subset for CI; full set opt-in.** CI runs on the committed subset (sized so the whole vertical executes in minutes on CPU — target a few thousand arxiv papers with their induced citation subgraph, and a fixed Air Routes region or the full small graph). The full-scale run is the opt-in workflow.

## `load_air_routes()`

Downloads (checksum-gated) and registers two sources into the db:
- a **node source** (airports) with columns `code` (key), `desc`, `city`, `country`, `continent`, `lat`, `lon`, `elev`, `runways`, `longest`, `region`;
- an **edge source** for `route` edges `(src_code, dst_code, dist)` and one for the `contains` hierarchy `(parent, child)` — both are **declared edges** for tiers 01–02 (`route` for the route graph, `contains` for the hierarchy).
Returns the registered source names + the committed airport-id subset. Tenancy: load under a single tenant so the two-tenant rail can be demonstrated by loading a second region under a second tenant.

## `load_ogbn_arxiv(subset=...)`

Downloads (checksum-gated) and registers:
- a **node source** (papers) with `paper_id` (key), `title`, `abstract` (the text to embed), `subject` (the 40-class label), `year` (for the date-based split);
- an **edge source** `cite_edges (src_paper_id, dst_paper_id)` — the **declared citation graph** (the BYOG signal for tier-04).
`subset` selects a connected subgraph of N papers (induced citation edges) using the committed ID list. Returns source names + the committed subset + the canonical train/val/test split (date-based) restricted to the subset.

## Licensing / redistribution

Both are publicly redistributable (ogbn-arxiv: ODC-BY; Air Routes: permissive from `krlawrence/graph`). Record the licenses in the loader docstrings and the repo `NOTICE`. Do not commit the full datasets — commit only the small subset artifacts + the ID lists; the loader fetches the rest on demand (checksum-gated).

## Success criteria

`load_air_routes()` and `load_ogbn_arxiv(subset=...)` register their sources, return the committed subsets, and are deterministic across runs (same committed IDs). `data/ids/*.txt` committed. Checksums enforced (a tampered/changed download fails). Licenses recorded. CI `--execute` of a trivial smoke notebook that loads both passes.
