# KV-arxiv — the ogbn-arxiv vertical (keystone / de-risk slice)

**Status:** spec — draft (hand-off). **The keystone — build this third (after K1, K2), before everything else.** One PR (it is one serial pipeline, not four chapters).
**Why it is the keystone:** it is the only vertical that reaches a *credible* tier-04, it validates every `jammi_ai` API call end-to-end, and it **emits the committed artifacts + `golden_metrics.json`** that K0's golden-sample layer needs and that every later chapter authors against. Until this is CI-green, nothing else starts.

**Build it as ONE serial pipeline.** The tiers chain through artifacts (embeddings → graph → propagated → fine-tuned model → context predictor → conformal). Author the tier *sections* in order; do not split into parallel chapters.

---

## Pipeline (tier sections, serial; each is a recipe in the K0 template)

Dataset: `load_ogbn_arxiv(subset=...)` (K2) — papers (title+abstract+subject label+year) + the declared `cite_edges`. Determinism contract (K0 §3) applies throughout: `exact=True`, committed IDs, pinned model, tolerances.

### Tier 01 — Construct
- **Substrate:** `generate_embeddings(source=papers, model=<modernbert>, columns=["title","abstract"], key="paper_id")` → `arxiv.embeddings`; then `build_neighbor_graph(arxiv.embeddings, k=…, exact=True)` → `arxiv.neighbor_graph`. Also register the external `cite_edges` (declared) — the two graphs to contrast.
- **Theory:** graph construction / topology from data (monograph **Part I**); kNN/similarity-graph construction.
- **Bridge note:** the *derived* similarity graph (S9 self-kNN) is the commodity; the *declared* citation graph carries signal embeddings can't reconstruct (the circularity contract). This contrast seeds the whole book.
- **Rail (provenance):** the neighbor-graph edges carry `similarity`; record where each edge came from (derived vs declared).
- **Measure:** intrinsic graph stats (degree dist, homophily of `cite_edges` vs `neighbor_graph` on the `subject` label) — the homophily number motivates tier-04.
- **Emits:** `arxiv.embeddings`, `arxiv.neighbor_graph` (committed).

### Tier 02 — Analyze
- **Substrate:** `db.propagate_embeddings(source=papers, embedding_table=arxiv.embeddings, edge_source=cite_edges, edge_src_column, edge_dst_column, direction="out", hops=2, weighting="degree_normalized", alpha=0.1)` → `arxiv.propagated`. Also show `weighting="uniform"` (random-walk mean) and `output="jumping_knowledge"` as the spectrum. Plus `search`/`assemble_context` to compare retrieval over propagated vs raw embeddings.
- **Theory:** graph signal processing — low-pass filtering on the graph (monograph **Part II**) = SGC/APPNP decoupled propagation (modern canon).
- **Bridge note (a signature chapter — coordinate with K-bridge):** **propagation = one low-pass graph filter = one SGC/APPNP layer = the vector-agg over neighbours.** One equation, three names, one `propagate_embeddings` call — and the `weighting` knob *is* the method choice: `uniform` = random-walk mean (SGC-flavoured), `degree_normalized`+`alpha` = symmetric-normalized with teleport (**APPNP**), `output="jumping_knowledge"` = concatenate-all-hops.
- **Rail (measurement):** does propagation help retrieval? `eval_compare(embedding_tables=[arxiv.embeddings, arxiv.propagated], …)` → recall delta with significance.
- **Emits:** `arxiv.propagated` (committed).

### Tier 03 — Learn
- **Substrate:** `fine_tune_graph(node_source=papers, id_column="paper_id", text_column="abstract", edge_source=cite_edges, src_column, dst_column, base_model=<modernbert>, edge_provenance="declared")` → `arxiv.ft_model`. (Contrast against base + against `edge_provenance="similarity"` to demonstrate the circularity contract empirically: declared edges give real gain, similarity edges ~none.)
- **Theory:** representation learning on graphs — node2vec/contrastive (monograph **Part III**; Hamilton).
- **Bridge note:** graph-supervised metric learning = contrastive fine-tune over walk-sampled declared-edge pairs.
- **Rail (measurement):** `eval_compare([base, ft_declared, ft_similarity], …)` → the declared-beats-similarity gap (the circularity contract, demonstrated not asserted).
- **Emits:** `arxiv.ft_model` (committed checkpoint).

### Tier 04 — Predict & Quantify (THE CRUX)
- **Substrate:** `train_context_predictor(source=papers, key_column="paper_id", task_column=<subject-or-cohort>, value_column=<target>, architecture="attncnp", output="gaussian", objective="crps", seed=0)` → `arxiv.ctx_predictor`; then `predict_with_context_predictor(arxiv.ctx_predictor, source=papers, target_key=…, edge_source=cite_edges, edge_hops=…)` — **graph-conditioned** prediction (the BYOG payoff: condition on the citation neighbourhood, not embedding-similar rows).
- **The crux sequence (the reason this dataset is the slice) — must run and measure:**
  1. **Marginal conformal** via the real API (`conformalize` for the classification set, or `conformalize_cqr`/`conformalize_interval` for the regression target), with a held-out calibration split.
  2. **Show it under-covers on the citation graph** — realised coverage on the graph-conditioned predictions falls below nominal (graph correlation breaks exchangeability — Barber 2023). This is `golden_metrics.tier04.marginal_coverage` (≈ below 1−α).
  3. **Apply the graph-aware repair INLINE in the notebook** — ~10 lines of numpy: locality-weighted conformal (≈ NAPS) — weight calibration nonconformity scores by graph proximity (or stratify by a citation-component cohort), recompute the finite-sample quantile. *This is the consumer making the cohort/weight choice — exactly where the doctrine puts it.*
  4. **Show coverage restored** — `golden_metrics.tier04.repaired_coverage` (≈ ≥ 1−α), with mean set-size/interval-width reported (sharpness).
  5. **Point to enterprise E8** as the productionized governed version (cite `20-enterprise/E8-graph-conditioned-governance.md`) — do **not** call/expect Mondrian/weighted from OSS Python (K0 §4).
- **Theory:** Neural-Process / context-conditioned posterior (Garnelo CNP; Kim ANP; the TabPFN/PFN line); conformal coverage + its break under graph dependence (Vovk; Angelopoulos & Bates; Barber 2023; CF-GNN / NAPS).
- **Bridge note (a signature chapter):** a prediction = a context-conditioned posterior; the context = a `search`/walk; the honest coverage statement requires the consumer to own the cohort choice — the tier with **no Neptune analogue**.
- **Rail (provenance):** the `source` fact + `context_ref` (which rows informed the prediction) ride every output — the audit trail for the coverage claim.
- **Rail (measurement):** `eval_calibration(shape="gaussian", …)` → CRPS/NLL/ECE/coverage/sharpness (R2).
- **Emits:** `arxiv.ctx_predictor`, `arxiv.cal_split`, and the four tier-04 golden metrics.

## Emit the cache (the keystone deliverable)

On a successful end-to-end run, **commit** to `artifacts/arxiv/`: `embeddings.parquet`, `neighbor_graph`, `propagated.parquet`, the fine-tuned checkpoint (or its id+hash), `cal_split`, and `golden_metrics.json` (all metrics above, each with a tolerance). These ARE K0's golden-sample layer; KV-air / K-rails / K-bridge load them and never recompute.

## Determinism contract

Embed K0 §3 verbatim. Specifically: `exact=True` neighbor graph + tie-break by `_row_id`; committed subset IDs; pinned modernbert + dtype + `OMP_NUM_THREADS=1`; CI loads the committed checkpoint (never retrains); the tier-04 calibration split committed (calibration disjoint from train **and** test); metrics asserted to tolerance; the inline repair seeded.

## Success criteria

The full vertical runs `--execute` CI-green on the committed subset; every tier ends in a real measured number matching `golden_metrics.json` within tolerance; **the tier-04 crux demonstrably shows under-coverage → repair → restored coverage** (the two coverage golden metrics straddle 1−α); the committed cache is populated; no band-aid tell-signs; every API call matches the K1 grounded reference; the four bridge notes are present (full prose deepened by K-bridge). This being green is the gate that unblocks KV-air / K-rails / K-bridge.
