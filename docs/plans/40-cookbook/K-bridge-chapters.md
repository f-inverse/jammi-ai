# K-bridge — the bridge chapters + the citation map

**Status:** spec — draft (hand-off). Build last; authored **against the committed cache** (uses golden metrics for any "measured" claim). One PR.
**Purpose:** the book's *thesis* made explicit — the signature move that **one Jammi recipe = one equation in the monograph = one line in the GNN canon** — plus the Neptune-contrast framing and the **citation map** that pins every recipe to its source equations. The verticals (KV-arxiv, KV-air) carry short "bridge notes"; this spec writes the full bridge chapters and the map they reference.

---

## The signature chapters (the equation = aggregate moves)

Each is a short chapter that takes one operation and shows it is the same object at three altitudes, then runs it:

1. **Propagation = low-pass graph filter = SGC/APPNP layer = the vector-agg over neighbours.** Monograph Part II (graph signal processing, the Laplacian low-pass) ↔ SGC/APPNP (Wu 2019; Klicpera/Gasteiger APPNP 2019) ↔ `propagate_embeddings`. The headline chapter — show that the `weighting` argument *is* the choice of filter: `uniform` = the random-walk operator `D̃⁻¹Ã` (SGC-flavoured), `degree_normalized`+`alpha` = the symmetric-normalized propagation with teleport restart (**APPNP**'s `(1−α)Â + α·X⁽⁰⁾`), `output="jumping_knowledge"` = stacking the filtered signals. One equation, three names, one call — and note its **byte-identical determinism** (fixed f64 fold order, no seeding) as the reproducible point on the propagate/learn spectrum. Measured on the cached arxiv artifacts.
2. **An edge is a self-`search`; a context set is a `search`/walk.** S9 self-kNN (construct) and S16/S16-G context assembly ↔ kNN graph construction (Part I) ↔ the retrieval-conditioned context of a Neural Process (CNP). The "retrieval is the substrate of both graphs and prediction" chapter.
3. **Graph-supervised metric learning = contrastive fine-tune over declared-edge walks.** node2vec/DeepWalk + contrastive (Part III; Hamilton) ↔ `fine_tune_graph(edge_provenance="declared")`. Include the circularity contract empirically (declared beats similarity), from the cached tier-03 metrics.
4. **A prediction is a context-conditioned posterior; honest coverage needs the consumer to own the cohort.** The Neural-Process family + conformal-under-graph-dependence ↔ tier-04. The chapter with **no monograph or Neptune analogue** — the moat. Reuses the KV-arxiv crux (marginal under-covers → inline repair → restored) as the worked example.

## The Neptune-contrast framing chapter

The book's organizing conceit: Neptune is **Database / Analytics / ML**; this cookbook is **Construct / Analyze / Learn / Predict & Quantify** — the same "unified yet separate" spine, re-expressed as runnable Jammi computation, **plus a fourth tier Neptune structurally lacks** (calibrated, provenance-stamped, context-conditioned prediction). State plainly: tiers 01–03 are roughly Neptune-parity (commodity); tier 04 + bring-your-own-graph is the non-redundant core. Name where Neptune uses Air Routes (queries/algorithms) and where it switches datasets for ML (so do we → ogbn-arxiv).

## The citation map

A table (rendered in the book; source-of-truth for the bridge notes) mapping each recipe → its monograph equation/section AND its GNN-canon reference. Representative rows:

| Tier / recipe | Monograph (Stanković et al. 2020) | GNN canon | Jammi call |
|---|---|---|---|
| 01 construct (kNN graph) | Part I (graph construction, Laplacian) | — | `build_neighbor_graph` |
| 02 propagation | Part II (low-pass graph filter / GFT) | SGC (Wu 2019), APPNP (Klicpera 2019) | `propagate_embeddings` |
| 03 graph-supervised | Part III (graph embeddings) | node2vec (Grover 2016), Hamilton 2017 | `fine_tune_graph` |
| 04 context posterior | — | CNP (Garnelo 2018), ANP (Kim 2019), TabPFN (Hollmann 2022) | `train_context_predictor` |
| 04 conformal + graph break | — | Vovk; Angelopoulos & Bates 2021; Barber 2023; CF-GNN (Huang 2023); NAPS (Clarkson 2023) | `conformalize*` + inline repair |

The executing agent **researches and verifies** every citation during the per-spec pressure-test (don't trust this table's specifics blind — confirm authors/years/venues), and expands the map to cover every recipe in the verticals.

## Determinism

Bridge chapters that show a number read it from the cache + assert against `golden_metrics.json`; no recomputation.

## Success criteria

The four signature chapters + the Neptune-contrast chapter render and `--execute` green against the cache; the citation map is complete (covers every recipe), verified, and is the source the verticals' bridge notes cite; the thesis ("one recipe = one equation = one canon line") is demonstrated, not merely asserted, on at least the propagation and tier-04 chapters; no band-aids.
