# EXECUTION-STATUS

Build state for the Theory↔Computation Cookbook (spec set: `jammi-ai`
`docs/plans/40-cookbook`). Mirrors the structure of the main tracker: a per-spec
status table, a decisions log (every fork resolved + every cut with rationale),
and a per-branch audit history.

## Per-spec status

| Spec | Status | Branch | PR | Notes |
|---|---|---|---|---|
| K1 scaffold + lib | merged | `k1-k0-scaffold` | #1 | repo skeleton, shared lib, CI harness, CLAUDE.md, grounded API reference |
| K0 contracts | merged | `k1-k0-scaffold` | #1 | folded into the K1 PR: `contracts.py` (2-layer registry) + `determinism.py` + philosophy gates in CLAUDE.md |
| K2 datasets | merged | `k2-datasets` | #2 | Air Routes + ogbn-arxiv loaders; committed subset IDs; datasets chapter |
| KV-arxiv keystone | merged | `kv-arxiv` | #3 | full 01–04 vertical; cache emitted + golden metrics committed; re-pinned to `jammi_ai==0.26.2` (the A3 bidirectional win is real; tier-04 reframed honest) |
| KV-air | merged | `kv-air` | #4 | tiers 01–02 on-ramp + the tenancy showcase (catalog-listing + discriminator-column isolation + the global-source caveat) |
| K-rails | merged | `k-rails` | #5 | rails chapter (provenance/tenancy/measurement, first-class) + the closed eval loop asserting the full golden chain |
| K-bridge | merged | `k-bridge` | #6 | four signature chapters + the Neptune-contrast framing + the independently-verified citation map (references.bib + a dangling-cite gate + a test pinning each call to a real engine verb) |
| #45-A2 conformal vertical | merged | `v-conformal` | #7 | standalone conformal chapter (08): LAC/APS/RAPS sets + abs-residual/CQR intervals on the committed tier-04 cache, plus the score-aligned weighted-restore (the keystone's complement). All measured, asserted to golden. |
| #45-A3 calibration vertical | merged | `v-calibration` | #8 | calibration & uncertainty chapter (09): `eval_calibration(gaussian)` + a client-local PIT/reliability fold on the committed tier-04 gaussian predictions. Proper scores (CRPS/NLL), ECE, sharpness, PIT-KS — all read from the cache, asserted to golden, cross-checked against numpy closed forms. |
| #45-A1 fine-tune methods | merged | `v-finetune` | #9 | fine-tuning-methods vertical: cosent/mnrl/triplet/hard-negatives/matryoshka/fine_tune_graph measured side-by-side on ogbn-arxiv; per-method recall@10 frozen to golden; hard-neg OOM finding recorded at scale (≥1500 pairs). |
| #45-B retrieval + tenancy | merged | `v-tier-b` | #10 | two CPU verticals on the committed cache (no GPU): B1 retrieval/search (chapter 10) — dense vs hybrid vs RRF-fused recall@10/nDCG@10 + the honest fusion finding (fusion does NOT help) + the search-multi-table engine finding; B2 multi-tenancy-as-measured-property (chapter 11) — the true two-layer isolation model + the honest caveat, measured live as hard-zeros + a visible count + tenant-conditioned metric parity. |
| 0.26.4 re-pin + `with_tenant` migration | merged | `chore/repin-0.26.4-tenant-migration` | #11 | re-pin `jammi_ai==0.26.4`; migrate all `with_tenant(t)` call sites to `set_tenant` / `tenant_scope`; `check_api_reference.py` re-grounded to the 0.26.4 surface (36 surfaces: 34 REQUIRED verbs + 2 module functions). |
| C1 mutable companion tables / feature store | merged | `feat/c1-mutable-feature-store` | #12 | chapter 12: `create_mutable_table` / `sql(INSERT/SELECT)` / federating JOIN; feature = citation in-degree (graph-derived, committed parquet); `SUM(in_degree)` JOIN aggregate asserted against golden; `scripts/build_feature_store_cache.py` + `tests/test_feature_store_cache.py`. |
| C2 CDC with trigger topics | merged | `feat/c2-cdc-triggers` | #13 | chapter 13: `register_topic` / `publish_topic` / `subscribe_collect` (predicate-filtered, from_offset replay); 60 record-change events (add/update/remove ops on committed arxiv paper_ids); bounded replay-collect counts asserted against golden; `scripts/build_cdc_cache.py` + `tests/test_cdc_cache.py`. |
| Eval on the wire + provenance channels (the ch05 R1 close) | merged | `feat/ch14-eval-channels` | #15 | chapter 14: the eval verb family (`eval_embeddings` / `eval_per_query` / `eval_inference` [classification + NER] / `eval_compare`) and the evidence-provenance channel registry (`register_channel` / `add_channel_columns` / `list_channels`) — the verbs that landed on `RemoteDatabase` via engine PRs #171/#172 — validated cross-transport. `scripts/build_eval_cache.py` runs every verb + the channel sequence on BOTH the embedded engine and a live remote `grpc://` `jammi-server`, asserting `remote == embedded` live (1e-9, NER `confidence` at its true f32-relative 1e-6) and recording the verdict in `artifacts/eval/eval.json`; committed embedded-canonical reports + frozen `golden_metrics.json` + `checksums.json`. PR CI reads the cache (aggregates-to-golden + shape) and runs the channel half live-embedded (#170 tenant-isolation / non-collision). Measured: embeddings recall@k 0.2833 / mrr 0.2722 / ndcg 0.2262; classification acc 0.4 / f1 0.2857; NER p/r/f1 0.0 (random-init fixture — well-defined rates); `eval_compare` self-anchor max\|Δ\| 0.0 + two-table (tiny_modernbert vs tiny_bert) recall Δ +0.2333 (p 0.2858, reproducible CI). `scripts/build_eval_cache.py` + `tests/test_eval_cache.py` + `tests/test_channels.py`. Pinned `jammi_ai==0.26.5` (the release that ships these verbs). |
| Scale tier — 170k arxiv ANN-vs-exact recall (H2 W1) | merged | `feat/scale-tier-arxiv-emit` | #16 | chapter 14-scale (`scale.qmd`): the scale tier on a committed 170k-row ogbn-arxiv subset (`data/ids/arxiv_scale.txt`); a frozen `usearch` ANN index (`artifacts/scale/arxiv_ann.usearch`, pinned `usearch==2.25.1`) cross-checked against exact search — ANN-vs-exact recall@k at scale asserted to golden. `scripts/build_scale_cache.py` + the `contracts` scale-artifact loaders; `artifacts/scale/golden_metrics.json` + `checksums.json` committed. |
| Scale ch14 — recall-vs-cost trade + usearch-drift close | merged | `feat/scale-recall-vs-cost` | #17 | extends 14-scale: the recall-vs-cost trade across ANN build/query params (`artifacts/scale/recall_sweep_build.json`), and closes the usearch-version drift (the `usearch.__version__`-vs-manifest assertion before any recompute). New scale goldens asserted; `checksums.json` updated. |
| Step-0 reconcile + `jammi_ai==0.29.0` re-pin | merged | `chore/reconcile-0.29.0-upgrade` | #18 | the reconcile ahead of the regression chapter: `pyproject` pin `0.26.5 → 0.29.0` (adds the scale tier + the scale-robust regression fine-tune surface); `_api_reference.md` re-introspected against the 0.29.0 wheel — `fine_tune`'s regression surface added (`seed`, `regression_loss`, `regression_beta`, `quantile_levels`) with the precise beta-NLL framing (stop-gradient re-weighting that prevents mean-starvation, distinct from the std floor); EXECUTION-STATUS flipped #15 → merged and added #16/#17. Gate green at 0.29.0: api-ref **40 surfaces**, ruff, pytest **92** (no conformal/search golden drift across the six-version jump), no-deferral, check_citations **32**, quarto render **19 chapters**. |
| Chapter 15 — regression fine-tune on a high-offset target (H2 W5 loop-closer) | merged | `feat/finetune-regression-chapter` | #19 | chapter 15: the public `fine_tune(task="regression")` surface measured — predicts ogbn-arxiv `year` (~2018, high-offset) from title+abstract across all four `regression_loss` objectives (`beta_nll`/`gaussian_nll`/`crps`/`pinball`), inferred on a seeded representative held-out split, folded to held-out RMSE-in-years + nominal coverage from the committed de-standardized predictions (every metric **re-folded** by the chapter/test, never hand-written). Honest measured findings: (1) the high-offset target **FITS WITHOUT COLLAPSE** — Gaussian heads serve a real spread (min `std_mean` 0.75y) vs the documented pre-0.26.2 `std≈0.001` degenerate failure — the **v0.29.0 z-space-loss win, measured**; (2) `year` is low-signal from text — every objective regresses to the conditional mean (RMSE ≈ test std ≈ 1.05y), loss choice a near-tie (crps marginally best); (3) Gaussian intervals mildly overconfident (coverage 0.736 < 0.90) and the pinball quantile band collapses (width 0.215y → 0.000 coverage) — a candidate engine observation, recorded honestly, not tuned away. GPU emit `scripts/build_finetune_regression_cache.py` (0.29.0-guarded, seed-recorded) + `tests/test_finetune_regression_cache.py` + 5-artifact contracts registry + `references.bib` (`nix1994`/`seitzer2022`/`koenker1978`). Gate green: api-ref **40**, pytest **97**, no-deferral, check_citations **35**, quarto render **20 chapters**. |
| Chapter 17 — the channel error taxonomy, each failure → its typed gRPC code (H3 §3.8) | merged | `feat/h3-cookbook-chapters` | #21 | chapter 17 (`17-channels-taxonomy/channels-taxonomy.qmd`): the engine `§3.8` channel error taxonomy (engine `#193`) MEASURED — each evidence-channel failure (`register_channel`/`add_channel_columns`) maps to its **CORRECT typed gRPC status code** on the wire instead of `Internal`-for-everything, validated cross-transport. `scripts/build_channels_taxonomy_cache.py` drives each failure mode on BOTH the embedded engine and a live remote `grpc://` `jammi-server`, measures the wire `StatusCode` on the `grpc://` arm (where the codes exist) with the embedded normalized error CLASS as the companion, asserts each maps as `#193` intended + `remote == embedded` on the class live; committed `artifacts/channels/matrix.json` + `golden_metrics.json` + `channels_taxonomy.json` + `checksums.json`. **CPU/hermetic** — channel ops are catalog ops, no GPU. **Measured taxonomy (all to golden, tol 0, ZERO deviation from #193):** duplicate registration → `ALREADY_EXISTS`; op on an unregistered channel → `NOT_FOUND`; column-redeclare conflict (different dtype) → `FAILED_PRECONDITION`; empty channel id → `INVALID_ARGUMENT`; **none collapses to `INTERNAL`/`UNKNOWN`**. `INTERNAL` recorded as the documented residual (a genuine DB fault is NOT fabricated). Honest nuance measured: an invalid dtype STRING is a CLIENT-SIDE `ValueError` (never reaches the wire), so the wire `INVALID_ARGUMENT` cell is an empty channel id (server-rejected). `remote == embedded` class for all 4 modes. `scripts/build_channels_taxonomy_cache.py` + `tests/test_channels_taxonomy.py`; `references.bib` (`grpcstatuscodes`, verified vs grpc.io). Gate green: api-ref **43**, ruff, pytest **111**, no-deferral, check_citations **36**; quarto via CI. |
| Chapter 16 — the model catalog, measured remote == embedded (H3 §3.6) | merged | `feat/h3-cookbook-chapters` | #21 | chapter 16 (`16-lifecycle/lifecycle.qmd`): the engine `§3.6` model-catalog surface (`list_models`/`describe_model`/`delete_model`, on BOTH the embedded `Database` and the remote `RemoteDatabase`) MEASURED as the **referential-integrity matrix**, validated `remote == embedded` for every observable. The catalog lets you **see** the models the engine resolves and trains, and **clean them up**; pre-trained models are served by **id**. `scripts/build_lifecycle_cache.py` runs the whole catalog interaction on BOTH the embedded engine and a live remote `grpc://` `jammi-server` and asserts parity live; committed embedded-canonical `artifacts/lifecycle/matrix.json` + `golden_metrics.json` + `lifecycle.json` + `checksums.json`. **CPU/hermetic** — registration via a tiny CPU `fine_tune` (the only public path that puts a model row in the catalog), candle falls back off CUDA. Measured matrix (all to golden, tol 0): register→`registered` (2 rows: base + fine-tuned), reflected by describe+list as the minimal projection `{model_id, backend, task, status}`; delete-referenced→typed `referenced` (FAILED_PRECONDITION on wire); delete-absent-strict→`not_found` (the typed **ModelNotFound** — NOT invalid-argument); delete-absent-`if_exists`→no-op; `every_catalog_model_is_referenced=True`; **`remote == embedded` for all 8 observables**. Measured catalog property: every model in the catalog is trained-and-referenced, so there is no bare unreferenced model to delete (the delete-unreferenced-succeeds path simply does not arise) — a property of the engine's catalog, not a gap. `scripts/build_lifecycle_cache.py` + `tests/test_lifecycle_cache.py`. Gate green: api-ref **43**, ruff, no-deferral, check_citations; quarto via CI. |
| Chapter 18 — per-verb tenant isolation + the BYO-auth seam (H3 §3.5) | merged | `feat/h3-cookbook-chapters` | #21 | chapter 18 (`18-tenancy-h3/tenancy-h3.qmd`): the engine `§3.5` **standing isolation oracle** MEASURED per-verb from the consumer side as **hard zeros**, plus the **BYO-auth seam** as a consumer-side worked example. Extends ch11's two-layer model to the full tenant-scoped verb surface. `scripts/build_tenancy_h3_cache.py` drives every verb on the embedded engine + (for the wire verbs) a live remote `grpc://` `jammi-server`, asserts `remote == embedded` live for all 10 cross-transport observables; committed `artifacts/tenancy_h3/matrix.json` + `golden_metrics.json` + `tenancy_h3.json` + `checksums.json`. **CPU/hermetic** — isolation is catalog/SQL behavior, no GPU. Reuses `rails.tenant`/`assert_listing_isolated`/`assert_rows_isolated` verbatim. **Measured matrix (all to golden, tol 0): 7 HARD ZEROS — `list_sources`/`describe_source`/`list_mutable_tables`/`list_topics`/`list_channels`/`list_models`/`sql` (discriminator-column Flight SQL row read) all leak 0 of A's resource to B.** Two STATED-POSITIVES (honest, not hidden): B sees the 3 built-in global channels (`vector`/`inference`/`bm25`); A reads a discriminator-less source whole (3 rows) — the engine does not authenticate. COLLISIONS never clobber: a duplicate `create_mutable_table` name across tenants ERRORS on the global PK; duplicate `register_topic`/`register_channel` ids isolate per-tenant (B's own). DESTRUCTIVE verbs are tenant-scoped: A's mutable table + topic SURVIVE B's `drop_mutable_table`/`drop_topic` (no cross-tenant destruction). **NO LEAK FOUND** — and notably the `drop_mutable_table` cross-tenant-destruction defect flagged in H3 scouting is **NOT present on the pinned 0.30.0 engine** (B's drop resolves in B's own namespace, A's table survives, measured). **BYO-auth seam (Part B):** a generic HMAC-SHA256 signed bearer token + an `AuthGateway` verifying it and binding the engine's `tenant_scope` IN FRONT of the verb — two authenticated tenants get isolated reads, a missing credential is rejected (not run unscoped), an invalid/forged credential is rejected (legit-same-tenant still resolves). NO real IdP, no product name — mirrors the engine's `grpc_byo_auth.rs` seam. `scripts/build_tenancy_h3_cache.py` + `tests/test_tenancy_h3_cache.py`; `references.bib` (`rfc2104` HMAC, `rfc6750` bearer-token — both RFC-Editor verified). Gate green: api-ref **43**, ruff, pytest **120**, no-deferral, check_citations **38**; quarto via CI. Names-no-consumer confirmed: opaque tenant UUIDs, generic verbs, no tenant/consumer vocabulary. |
| H4 PR #0 — `jammi_ai==0.31.0` re-pin + broken-venv fix | in progress | `h4-pin-0.31.0` | — | the H4-batch reconcile: a **pure regression re-pin** (like #18), ZERO golden drift. (1) **Venv fix** — the repo `.venv` had a LOCAL EDITABLE `jammi_client` pointed at the engine repo `clients/python` shadowing the published wheel, its protobuf stale (`pipeline_pb2` had no `Cascade`), so even `import jammi_ai` failed; recreated `.venv` clean and `pip install -e ".[book,dev]"` so the PUBLISHED `jammi_ai==0.31.0` and its **vendored** (non-editable, site-packages) `jammi_client==0.31.0` install — no editable engine client on the path; `import jammi_ai, jammi_client` + `from jammi_client import RemoteDatabase` confirmed, all 5 H4 verbs present on a live `Database`. (2) **Pin** `pyproject` `0.30.0 → 0.31.0` (+ comment block naming the H4 verbs); `jammi-client` NOT separately pinned (vendored transitively); README dev-line corrected to `==0.31.0`. (3) **API guard re-introspected** against the 0.31.0 wheel — the 5 H4 verbs added (`asof_join` [spine_by/spine_time/facts_by/facts_time/direction/boundary/tolerance_duration_micros/tie_break_column/project], `verify_materialization` [expected_definition], `staleness` [current_definition], `derives_from` [none], `recompute` [cascade]) + the `cache` kwarg on `generate_embeddings`/`infer`/`build_neighbor_graph`/`propagate_embeddings`; guard **43 → 48 surfaces**; `_api_reference.md` re-grounded (header bumped, cache kwarg added to the 4 verbs, new H4 point-in-time + materialization section, every signature copied from the live wheel). Gate: api-ref **48 surfaces** (see below). No new chapters. |

## #45 expansion verticals — measured goldens

### B1 retrieval / B2 tenancy (`v-tier-b`, chapters 10–11)

**B1 retrieval (CPU, committed cache, same-subject golden, 200 queries):**

| retriever | recall@10 | nDCG@10 |
|---|---|---|
| dense raw | 0.538 | 0.547 |
| dense propagated | 0.556 | 0.563 |
| lexical BM25 (titles) | 0.417 | 0.369 |
| RRF(raw+prop) | 0.550 | 0.559 |
| hybrid RRF(raw+lex) | 0.542 | 0.556 |
| RRF(prop+lex) | 0.529 | 0.551 |

The **honest fusion finding (fusion does NOT help)**: RRF of the two dense arms beats the
*weaker* arm (raw, +0.012) but sits *below* the *stronger* single arm (propagated,
−0.006); the hybrid dense+lexical fusion edges past raw (+0.004) but is dragged well below
propagated by the weaker lexical ranker. No fused list beats the best single arm
(`fusion_helps=False`). RRF cannot exceed the best single ranker it already contains when
one ranker dominates — graph propagation, not fusion, is the lever on this target. Dense
reproduces the keystone's per-table recall exactly (raw == tier01 0.538, propagated ==
tier02 0.556).

The **search-multi-table engine finding** (handled + recorded, a candidate the user files):
`search(source, *, query, k, filter, select)` resolves a source's single ready embedding
table and carries **no `table=` argument**. A committed embedding parquet is not a
source-bound ready embedding table (`search` / `assemble_context(hybrid=True)` raise
`No ready embedding table for source`), and once a source has several embedding tables
(raw / propagated / fine-tuned) there is no way to name which to search — ambiguous. The
proven measurement is the keystone's exact cosine-kNN numpy fold (deterministic,
table-exact); `rrf_fuse` IS reachable on the embedded CPU handle and is reused directly.
Candidate: an explicit `table=` argument on `search`.

**B2 tenancy (CPU, committed cache, the true engine model measured live):**

| property | value | role |
|---|---|---|
| catalog-listing leak | 0 | hard zero (A's `list_sources` excludes B's source) |
| discriminator-column leak | 0 | hard zero (one tagged source, disjoint rows under A vs B) |
| discriminator-less caveat visible | 1451 | positive (A sees ALL of B's rows — the honest limit) |
| parity recall A | 0.756 (2549 rows / 113 q) | tenant-conditioned scoped result |
| parity recall B | 0.698 (1451 rows / 87 q) | tenant-conditioned scoped result |

The two isolation layers are HARD zeros; the discriminator-less caveat is a POSITIVE
assertion (a separate source is hidden from the LISTING, not from a direct named read — the
KV-air correction, not re-broken). Tenant-conditioned metric parity: the same recall recipe
under two tenants yields each its own scoped result over a disjoint partition that tiles the
cache (2549 + 1451 = 4000). Reuses `rails.tenant` / `assert_listing_isolated` /
`assert_rows_isolated` verbatim. New cites verified: RRF (Cormack, Clarke & Büttcher, SIGIR
2009) and BM25 (Robertson & Zaragoza, FnT IR 3(4), 2009).

## #45 expansion verticals — measured goldens

### A2 conformal (`v-conformal`, chapter 08)

Per-score, on the committed tier-04 cache (CPU, no recompute), α = 0.10:

| score | coverage | mean set size / width |
|---|---|---|
| LAC | 0.889 | 7.97 |
| APS | 0.867 | 6.17 |
| RAPS | 0.867 | 6.17 (== APS — penalty does not bite at this class count) |
| abs-residual interval | 0.830 | 1.93 years |
| CQR (±1σ band) | 0.874 | 2.34 years |

Honest measured deviations from the textbook ideal (reported, not tuned away):
**RAPS reduces to APS** on this cache for every `(λ, k_reg)` tried; and **APS, not
LAC, is the sharper family here** (LAC reaches higher realised coverage with larger
sets — the "LAC = smallest" result holds only at exact nominal calibration, not under
this time-split). Every family **under-covers** the nominal 0.90 (non-exchangeability
is real).

The score-aligned weighted-restore (the keystone's complement), a **transparently-
synthetic, clearly-labelled** teaching device (seed 0, γ = 1.2, |cal| = 500):

- marginal split-conformal coverage **0.833** (under-covers)
- weighted split-conformal coverage **0.939** (restored to ≥ nominal)
- `corr(nonconformity, shift-feature) = +0.730` (high, positive)

This is the explicit **inverse** of the keystone's tier-04 no-op (`corr ≈ −0.12`,
Δ ≈ 0). The thesis the two halves carry: **weighted conformal repairs a covariate
shift iff the shift moves the nonconformity-score distribution.** No claim is made
that the real ogbn-arxiv time-split is repairable by weighting — it is not.
| #45-A3 calibration vertical | in review | `v-calibration` | — | calibration & uncertainty chapter (09): eval_calibration(gaussian) + a client-local PIT/reliability fold on the committed tier-04 gaussian predictions. Proper scores (CRPS/NLL), ECE, sharpness, PIT-KS — all read from the cache, asserted to golden, cross-checked against numpy closed forms. |

## #45 expansion verticals — measured goldens

### A3 calibration (`v-calibration`, chapter 09)

`eval_calibration(shape="gaussian")` on the committed tier-04 gaussian year
predictions (test era, n = 2115), cross-checked against an independent numpy
closed-form fold:

| metric | value | role |
|---|---|---|
| CRPS | 0.386 | proper score (years; engine == manual closed form) |
| NLL | 1.076 | proper score (engine == manual closed form) |
| adaptive ECE | 0.160 | calibration diagnostic |
| sharpness | 2.935 | mean predictive σ (years) |
| central coverage | 0.952 | central-interval coverage |
| PIT KS | 0.417 | departure from Uniform(0,1) |

The honest finding: the predictor is **sharp but miscalibrated** under the time-split
— a narrow ≈ 2.9-year spread with a PIT far from uniform (KS ≈ 0.42) and a high ECE.
This is the same non-exchangeability the conformal tier reports as under-coverage,
read here as a non-uniform PIT. The proper scores (engine values reproduced exactly by
the gaussian closed forms) penalize the miscalibration; the PIT localizes it. The
principle — maximize **sharpness subject to calibration** — places this predictor on
the wrong side of the tradeoff, and the engine reports it rather than hiding it.

Fork note: `eval_calibration`'s `golden_source` resolves in the engine's default
catalog, so it is passed the **fully-qualified catalog path**
(`<source>.public.<source>`), not the bare source id. (Resolved from the installed
API — the bare id raises `table not found`.)
| A1 fine-tune methods | in review | `v-finetune` | — | the fine-tuning-methods vertical: cosent / mnrl (two temps) / triplet / hard-negatives / matryoshka / fine_tune_graph(declared) measured side-by-side on the committed ogbn-arxiv subset; emits `artifacts/finetune/` + per-method goldens; the honest finding (the supervision caps the gain, tier-03 circularity from the graph to the loss) |

## Decisions log

- **`§3.6` model-catalog chapter — the registration path + the referential matrix
  (2026-06-16).** The chapter MEASURES the engine's model catalog
  (`list_models` / `describe_model` / `delete_model`) cross-transport, `remote ==
  embedded`. The catalog lets you **see** the models the engine resolves and trains, and
  **clean them up**; pre-trained models are served by **id** (the resolver loads an HF or
  local model by reference), so a model enters the catalog by being trained. Three facts,
  all measured live and asserted to golden:
  - **Registration path.** The only public path that puts a model row in the catalog is
    **training** (introspected the wheel + the `jammi_client`: the model verbs are exactly
    `list_models` / `describe_model` / `delete_model`). A tiny CPU
    `fine_tune(..., method="lora", task="text_embedding")` over a 12-row in-memory
    `(anchor, positive)` pairs corpus with the engine's public `tiny_modernbert` fixture
    registers **two** model rows — the base model (path-keyed, `status="registered"`) at
    submission time, and the fine-tuned model (`jammi:fine-tuned:<uuid>`,
    `status="registered"`) on completion. This trains on CPU in seconds (the candle backend
    logs `CUDA requested … running on CPU`), so the chapter stays **CPU/hermetic** — no
    GPU, no keystone corpus. The model_id is a per-run UUID, so the committed cache freezes
    the matrix VERDICTS + observable counts, never the UUID.
  - **The referential-integrity matrix.** Every model in the catalog is
    trained-and-referenced: a fine-tuned model is referenced by
    `training_jobs.output_model_id`; its base model by `training_jobs.base_model_id`.
    `delete_model` on either raises the typed `ModelReferenced` error (embedded `RuntimeError
    "Model referenced: … still referenced by training_jobs.<edge>"`; wire
    `StatusCode.FAILED_PRECONDITION`). Because every model in the catalog is referenced,
    there is no bare unreferenced model to delete — the delete-unreferenced-succeeds path
    simply does not arise here. The chapter measures this as a real property of the engine's
    catalog (`every_catalog_model_is_referenced = True`); it is a property of the surface,
    not a gap.
  - **Error-surface parity is normalized, not literal.** The two transports raise different
    Python exception *types* by construction — embedded raises `RuntimeError` with a
    message; the gRPC client raises `grpc.RpcError` with a `.code()` `StatusCode`. The
    chapter's `remote == embedded` observable is the **normalized error class**
    (`referenced` / `not_found`) derived from each transport's native error
    (message-substring for embedded; `StatusCode.FAILED_PRECONDITION` → `referenced`,
    `StatusCode.NOT_FOUND` → `not_found` for remote). That normalization is the honest
    cross-transport contract for a typed error, the same shape the engine's own conformance
    pins (the wire `Model` projection `{model_id, backend, task, status}` agrees
    key-for-key; the *errors* agree class-for-class). Verified live: delete-absent (no flag)
    is `NOT_FOUND` on the wire (the typed `ModelNotFound` — **not** `INVALID_ARGUMENT`) and
    "Model not found" embedded; `if_exists=True` is a no-op on both.
- **H3 Step-0 reconcile (2026-06-15) — wave order + `§3.8` scope.** Re-baselined against
  engine `main` @ `f301969` (`py-v0.29.0-6`). Reconciled facts: `§3.8` channel-errors
  (`#193`), `§3.7` operability (`#194`–`#196`), and `§3.8` **transport-parity collapse
  (`#197`)** are all **landed on `main`, unreleased**. **Remaining engine H3 runs serial
  in the order `§3.6` lifecycle → `§3.5` multi-tenant → `§3.8` API-stability staging.**
  Two forks resolved: (1) **`§3.6` before `§3.5`** (corrects the earlier E2=`§3.5`/E3=`§3.6`
  labels) — binding because the `§3.5` standing isolation oracle must cover `§3.6`'s
  net-new lifecycle verbs (engine `H3-WAVE-PLAN.md` cross-workstream rule). (2)
  **`§3.8` API-stability staging (provisional/stable verb annotations + lifting
  `check_api_reference.py` into engine CI) lands LAST in H3**, over the feature-complete
  surface, so the annotation pass runs once rather than being re-touched after `§3.6`/`§3.5`
  add verbs; kept in H3 (not pushed to H4) per ROADMAP `§3.8`. The `§3.8`
  channel-status-codes cookbook chapter (validating `#193` cross-transport) can be authored
  against a local build now, independent of the remaining engine chain. Baseline green at
  Step 0: cookbook gate (api-ref **40** · ruff · pytest **97** · no-deferral **26** ·
  citations **35**; quarto via CI) + engine `cargo check --workspace --all-targets` clean
  on the host. Full sequencing: `H3-WAVE-PLAN.md`.
- **Repo target.** `f-inverse/jammi-cookbook`, **private**; created via `gh`.
  Book publishes to **GitHub Pages**. (User decision.) Caveat recorded: Pages on a
  private repo needs a Team/Enterprise plan — if unavailable, render-only until
  the repo is public; revisit at the release step.
- **Book tool: Quarto** (K1 allows Quarto or Jupyter Book; chose K1's primary
  suggestion). Recorded per the "pick one, record it" instruction.
- **`jammi_ai` re-pinned `==0.25.0` → `==0.26.0`** (the keystone slice). 0.26.0 is
  on PyPI with a `manylinux_2_28_x86_64` CPU wheel; the GPU compute tier is the
  published `jammi-server-cu12==0.26.0` server reached over `connect("grpc://…")`.
  The re-pin lands in `pyproject.toml`, `chapters/api-reference.qmd`, and the
  grounded reference. The reference + API guard were **re-grounded by introspecting
  the installed 0.26.0 wheel** (`Database` / `RemoteDatabase` / `jammi_client`),
  not transcribed.
- **0.26.0 control/data-plane split + `connect` parity (settled framing).** 0.26.0
  exposes `Database` (embedded, `connect("file://…")`) and `RemoteDatabase` (the
  pure-Python `jammi_client`, `connect("grpc://…")`) with an identical verb
  surface; `connect(target)` selects the transport once. The book's structural
  spine is that parity: write a recipe once, swap only `connect()` — `file://` for
  the CPU embed read-path, `grpc://` for the GPU compute emit. Chapters open
  `connect("file://…")` (replacing the old `open_local()`); the keystone emit opens
  `connect("grpc://…")` to the GPU server. GPU ML cannot run on the CPU embed
  wheel — it runs against `jammi-server-cu12`.
- **`fine_tune` signature drift corrected against 0.26.0.** The K2-era note said
  the wheel used `task=` *instead of* `method=`. 0.26.0's introspected signature
  has **both**: `method` is now a **required** keyword (the adapter family — only
  `"lora"` is accepted; a missing `method` raises `missing 1 required keyword
  argument: 'method'`), and `task` is the optional `ModelTask` string
  (`"text_embedding"` / `"classification"` / …). The grounded reference and the API
  guard now require `method`. The keystone uses `fine_tune_graph` (which carries no
  `method` kwarg), so this only affects the reference's `fine_tune` entry. (Fork
  resolved from the source of truth — the installed API.)
- **`build_neighbor_graph` takes the SOURCE name, not the embedding-table name.**
  Introspecting + running 0.26.0 showed `build_neighbor_graph(source, …)` expects
  the registered source id (e.g. `"arxiv_papers"`) and discovers that source's
  ready embedding table itself; passing the embedding-table name raises
  `No ready embedding table for source '<embtable>'`. The emit script and the
  grounded reference pass the source name.
- **Engine-produced tables are addressed `"jammi.<table>"`.** A registered file
  source is queried `<source>.public.<source>`; an engine result table (embeddings,
  neighbor graph, propagated) is a single quoted identifier `"jammi.<table>"`.
- **`search` / `eval_compare` target a *source*, so per-table recall is computed in
  numpy.** Once a source has several embedding tables (raw, propagated, fine-tuned),
  `search(source, …)` is ambiguous (it raised a hydration error). And at the time the
  keystone was built `eval_compare` / `eval_embeddings` were **not yet on
  `RemoteDatabase`** (only `eval_calibration` was) — they have since landed on the
  wire via engine PRs #171/#172 (validated end-to-end in **chapter 14**), but that
  does **not** retire the numpy fold here: the keystone's committed arxiv vectors are
  a **bare matrix**, not a source-bound engine embedding table, so even on the wire
  `eval_embeddings` could not score them (it needs a source-bound table + ANN
  sidecar). So the keystone measures per-table same-subject recall@10 with an exact
  cosine-kNN numpy fold over the committed embedding matrix — deterministic and
  table-exact. The recall numbers are real (raw 0.538 → propagated 0.556 →
  declared-edge fine-tune 0.548).
- **Tier-04 crux: classification (subject) under the time-split, NOT year
  regression — a build-or-cut.** The first emit ran the spec's regression option
  (`train_context_predictor(output="gaussian", value_column="year")` →
  `conformalize_interval`). On this recent 4000-paper subset `year` is degenerate:
  it spans only 2014–2020 and the gaussian predictor collapsed — `predict_with_
  context_predictor` returned `std≈0.001` and badly-biased means (~2167 for a
  2014–2020 target, residuals ~114y), so the interval over-covered (0.922) and the
  repair was a no-op (identical width). That is not a usable crux. **Cut the
  regression conformal path** (rationale: degenerate low-variance target →
  variance/mean collapse) and built the **classification** option the spec also
  offers and the graph-conformal literature (CF-GNN, NAPS) actually reports on
  ogbn-arxiv: a consumer nearest-centroid softmax over the **propagated** (graph-
  conditioned, tier-02) embeddings, the dataset's own **time-split** as the
  genuinely non-exchangeable calibration/test (earlier era → later era), marginal
  `conformalize(score="aps")` → it **under-covers (0.867)**, inline **weighted
  split-conformal** (Tibshirani 2019; weight calibration by test-era likeness in the
  graph-conditioned embedding) → **restored (≈0.895)**. `train_context_predictor` /
  `predict_with_context_predictor` are still exercised (the graph-conditioned
  posterior + the `source`/`context_ref` provenance rail), just not as the
  coverage substrate. This is the consumer owning the cohort/weight choice, the
  doctrine's exact placement.
- **0.26.2 RESTORES the year-regression conformal path — the A3 bidirectional win
  is real (supersedes the cut above).** The earlier cut blamed a *degenerate target*;
  the truer cause was the engine's **gaussian-collapse bug (#43)**. 0.26.1 fixed the
  fine-tune projection head but the *amortized context predictor* still collapsed;
  authoring this keystone surfaced that, and **0.26.2 completed it** (z-space
  standardization of the predictor's target + in-context members' y). Now
  `train_context_predictor(gaussian, value_column="year")` **fits** — cal-mean
  ≈2018.55, test-mean ≈2018.64, std ≈0.916 (NOT the old ~2163 / 0.001 collapse), so
  the previously-impossible regression-conformal workflow **runs end-to-end**. This
  is the headline: the cookbook→engine→cookbook loop, twice (#43 surfaced here →
  0.26.1 → re-emit surfaced the predictor still unfixed → 0.26.2). Both the
  regression path AND the classification path are now kept as tier-04's two cruxes.
- **The tier-04 conformal finding is reframed HONEST: both cruxes under-cover and
  weighting is a NO-OP in BOTH (no restore).** The earlier classification entry
  claimed a weighted *restore* (≈0.895). That was an APS-convention artifact, not a
  real repair: a local APS that didn't reproduce the engine's deterministic-APS
  admission rule. The decided, measured truth: the dataset time-split (cal=2018 era,
  test=2019/2020) is a **location/orthogonal shift, not a score-distribution shift**,
  so importance-weighted conformal cannot move the quantile. *Regression* — the
  predictor regresses to the embedding-conditioned mean (~2018.6 for both eras), so
  cal and test |y−ŷ| residual magnitudes are ≈equal (it's point-prediction *bias*,
  not a spread shift); corr(|residual|, test-likeness) ≈ 0; the test-likeness-
  weighted interval is an exact no-op (Δ +0.0000, still 0.830). *Classification* —
  the shift is ≈orthogonal to the APS score (corr(nonconformity, test-likeness)
  ≈ −0.12); the three weighting schemes move coverage by −0.001 / +0.022 / +0.006
  and NONE reaches nominal (best 0.889 < 0.90) — weighting does not repair (the
  movements are small and not even consistently toward nominal).
  The local-must-equal-engine `RuntimeError` guard is removed: the comparison runs
  on a single self-consistent local APS routine (apples-to-apples by construction),
  and the engine's marginal (0.867) is reported separately as the OSS-surface
  corroboration — both under-cover, the ≤~0.03 gap is a benign deterministic-APS
  set-boundary convention difference. The unifying lesson: weighted conformal
  repairs a covariate shift only when the shift moves the nonconformity-score
  distribution; a location/orthogonal shift needs a governed time-aware cohort. The
  score-aligned case where weighting genuinely repairs is forward-pointed to the
  planned conformal vertical (not built here).
- **`fine_tune_graph(epochs=1)` ran 3 epochs (observation; engine verified exact;
  the knob is dropped on the remote-client path).** During the 0.26.0-era kv-arxiv
  emit (over `grpc://`, `scripts/build_arxiv_cache.py` passing `epochs=1`) the
  server logged `Epoch complete epoch=0/1/2`. The original guess ("the trainer
  evidently has a floor or an internal schedule") was wrong. What is verified from
  the engine repo: (a) **the engine does not override `epochs`** — jammi-ai #160
  (0.26.4) concluded the #64 epoch question was *not a bug* (exactly one
  `for epoch in 0..epochs` loop shared by the tabular and graph paths, no
  warmup/zeroth epoch) and added a regression-guard oracle
  (`optimizer steps == epochs × ⌈batches/grad_accum⌉`); no commit between v0.25.0
  and v0.26.4 changes `epochs` handling in `fine_tune`/`fine_tune_graph`.
  (b) The observed 3 epochs matches the wire behavior readable in the client code:
  `RemoteDatabase.fine_tune_graph` (`clients/python/jammi_client/_database.py`)
  builds the proto `FineTuneConfig` — including `epochs` — but never attaches it to
  `StartTrainingRequest.config` (plain `fine_tune` does pass `config=config`), and
  an absent config resolves to the engine default `epochs: 3` server-side. That
  omission is present at the v0.26.0 tag and still present on jammi-ai `main` as of
  0.26.4. So the runtime expectation **still applies to a `grpc://` emit** (3
  epochs for `epochs=1`, ~10 min/epoch on the A10G) until the client attaches the
  config — reported upstream as `jammi-ai#167`. On the embedded
  path, and wherever the config reaches the engine, the epoch count is exact and
  oracle-pinned as of 0.26.4 (#160). No release in 0.25.0–0.26.4 is identified as
  having "fixed" an epochs override, because the engine never had one.
- **`datasets.py` deferred to K2, by capability not by limbo.** The K1 scaffold
  ships `contracts` / `determinism` / `rails`; the loaders are a distinct
  capability (K2) and land in their own PR. This is the split-by-capability rule
  (introduce the surface with no callers first), not a band-aid — the core lib
  imports clean without the loaders, and `pyproject.toml` gates the `ogb`
  dependency behind the `data` extra.
- **`ogb` is an opt-in extra, not a core dep.** It pulls torch/pandas; the core
  contracts/rails surface must import without it, so the loaders import it lazily.
- **No consumer names in the open-core book — the scrubbed references were
  removed (2026-06-16).** The open-core cookbook, like the engine, names no
  consumer; the earlier forward-reference framing was itself a consumer-name and
  was removed, since the open-core book names no consumer at all. The
  conformal chapter (ch08), `CLAUDE.md`, and `_api_reference.md` now state only what
  the OSS surface does — the marginal `conformalize*`, with the inline graph-aware
  repair as the user's own cohort/weight choice; a productionized governed cohort
  surface is out of scope and named nowhere.
- **K2: dropped `ogb`/`torch`, read `arxiv.zip` directly.** `ogb==1.3.6` is
  incompatible with `torch>=2.6` (its cached `.pt` fails the new
  `weights_only=True` default). Rather than pin an old torch or monkeypatch
  `torch.load` (band-aids), the loader reads the canonical edges/labels/year/
  time-split + id/label mappings straight from the pinned, checksum-gated
  `arxiv.zip` with the standard library. More deterministic (digest-pinned, no
  pickle cache), no heavy dependency. The `data` extra was removed; the loaders
  need only stdlib + pyarrow + requests.
- **K2 sources are checksum-pinned by content.** Air Routes pinned to commit
  `efd3b1ae…` (the moving `master` raw URL served stale CDN bytes — caught by the
  gate). arxiv.zip `49f85c80…`, titleabs `7bce99ab…`.
- **K2 subset selection.** ogbn-arxiv: a connected BFS ball from the highest-degree
  node, 4000 papers / 9595 induced citation edges, subject homophily **0.500**
  (random ≈ 0.025) and a populated time-split (train 804 / valid 1081 / test
  2115) — the precondition the tier-04 coverage crux needs. Air Routes: the full
  small graph (3504 airports), committed for the determinism contract.
- **K2 source query form.** A registered file source is queried as
  `<source>.public.<source>` (catalog.schema.table), not by bare name — noted for
  the keystone's tier chapters.

## Audit history

_(Populated before each merge: findings + remediations.)_

### `k1-k0-scaffold`
- **Independent audit (pre-PR), all green:**
  1. `jammi_cookbook` imports clean (applies the determinism env on import).
  2. API-reference guard — 24 `jammi_ai==0.25.0` surfaces confirmed against the
     installed wheel.
  3. `ruff check` clean.
  4. `pytest` — 11 unit tests pass (contracts registry + golden oracle;
     determinism seed/ids).
  5. No-deferral grep clean (scans the lib + every `.qmd`, including the root home
     page).
  6. `quarto render` executes the book end-to-end → `_book/` built.
- **Findings remediated before PR:**
  - ruff import-order in `rails.py` (auto-fixed).
  - no-deferral grep flagged the literal word "placeholder" in `index.qmd` prose;
    reworded to "stand-in figure" rather than weaken the gate.
  - Quarto book home page must be at the project root → moved `index.qmd` to root,
    fixed the API-reference cross-link.
  - Widened the no-deferral grep to scan root-level `.qmd` (the home page had
    escaped the `chapters/`-only scan).

### `k2-datasets`
- **Independent audit (pre-PR), all green:** api guard (24 surfaces) · ruff clean
  · pytest 16 pass (hermetic loader logic: checksum gate, connected-subset
  determinism + connectivity, schemas) · no-deferral clean (8 files) · `quarto
  render` executes the datasets chapter (7 cells with embedded assertions on real
  counts) end-to-end.
- **Validated against a real `jammi_ai` db:** both loaders register their sources;
  queries via `<source>.public.<source>` return the expected counts (airports
  3504 / routes 50637 / contains 7008; papers 4000 / cite 9595); ATL→continent NA;
  homophily 0.500.
- **Findings remediated before PR:** removed two band-aids that crept into the
  first loader draft (an unused `io` import with `_ = io`, a redundant `# noqa`
  pandas import); reworked the arxiv path off `ogb`/`torch` (see decisions log);
  `ruff format` for line-length.

### `kv-arxiv`
- **The GPU emit (the keystone deliverable), against the published
  `jammi-server-cu12==0.26.0` on the A10G** (`scripts/build_arxiv_cache.py --target
  grpc://127.0.0.1:50051`). `nvidia-smi --query-compute-apps` confirmed the server
  process held the device (ModernBERT embedding ~3.9 GB, fine-tune at 90–100% util).
  Emitted `artifacts/arxiv/`: `papers` · `embeddings` · `neighbor_graph` ·
  `cite_edges` · `propagated` · `ft_model.json` · `ctx_predictor.json` ·
  `cal_split.json` · `tier04_predictions` (f32) · `subject_golden` ·
  `golden_metrics.json` · `checksums.json` (~38 MB).
- **The real measured numbers (frozen as golden metrics):** tier01 cite-homophily
  **0.500** / neighbor-homophily 0.559 / recall@10 **0.538**; tier02 propagated
  recall **0.556** (Δ +0.018, low-pass denoising gain); tier03 declared-edge
  fine-tune recall **0.548** (Δ +0.010 over base, the circularity contract);
  tier04 (re-emitted at **0.26.2**) — the A3 bidirectional win: the gaussian
  context-predictor, which collapsed pre-0.26.2 on the low-variance `year` target
  (~2163 / std 0.001), now FITS (cal-mean **2018.55** / test-mean **2018.64** /
  std **0.916**). Both conformal cruxes UNDER-cover under the dataset time-split:
  year-regression interval **0.830** (nominal 0.90, width 1.93y), subject-
  classification APS **0.867** (set size 6.17). Importance-weighted conformal does
  NOT restore either — regression is an exact no-op (location shift: cal-residual
  0.560 ≈ test 0.543, corr +0.001); classification moves coverage but no scheme
  reaches nominal (kNN is the largest mover, +0.022 → 0.889, still under-covering;
  corr(nonconformity, test-likeness) −0.121). The honest lesson: weighted conformal
  repairs only a score-distribution-moving shift; this location/orthogonal shift
  needs a governed time-aware cohort, not a client-side reweight.
- **Independent audit (0.26.2 re-emit), all green:** ruff clean · API guard (26
  surfaces, **0.26.2**) · no-deferral clean (12 files) · pytest **24 pass** (cache-
  backed keystone tests asserting the *honest* properties — predictor fits ≈2018;
  both cruxes under-cover; no weighting scheme reaches nominal; the location- and
  orthogonal-shift diagnostics) · `quarto render` executes the whole book end-to-end
  on CPU against the committed cache, every golden assertion passing under real
  execution. The audit flagged + fixed a "no-op" overclaim (kNN closes ~2/3 of the
  classification gap) — prose + cache note reworded to "moves coverage but does not
  restore it"; the regression case stays a genuine exact no-op.
- **Findings remediated before PR:** (1) `recall_at_k` first used `db.search`,
  which targets a *source* and broke once a source had multiple embedding tables —
  reworked to an exact cosine-kNN numpy fold over the committed embedding matrix.
  (2) The `year`-regression tier-04 crux first failed (predictor mean/variance
  collapse) — which turned out to be engine bug #43, fixed across 0.26.1 (fine-tune
  head) and 0.26.2 (the amortized context predictor; surfaced by this very re-emit).
  At 0.26.2 the predictor fits, so the year-regression-conformal path is RESTORED as
  the tier-04 lead (the bidirectional win), with subject-classification kept as the
  honest under-shift contrast. The bug-fix loop, not a permanent cut. (3) `tier04_predictions.parquet` downcast to
  f32 (conformal result identical to four decimals — verified) to halve the
  committed size.

### `kv-air`
- **The on-ramp (tiers 01–02), clean and unchanged:** Air Routes — 3504 airports,
  the declared `route` and `contains` graphs — re-expressed as runnable Jammi
  computation. Tier 01 embeds airport text → neighbor graph and contrasts the
  declared `route` vs `contains` homophily on the continent label (route **0.826**;
  the containment hierarchy near-perfectly continent-consistent **0.99**, measured
  only over continent-parent edges to avoid a transcontinental-country artifact);
  recall@10 over raw embeddings **0.747**. Tier 02 propagates over the route graph
  (low-pass filter), lifting same-continent recall to **0.919** (Δ **+0.173**). The
  GPU emit ran end-to-end against `jammi-server-cu12==0.26.2` over `grpc://`; every
  chapter reads the committed `artifacts/air/` cache on CPU.
- **The tenancy showcase — reframed to the true engine model.** The original
  framing overclaimed a tenant-isolation property the engine does not provide: it
  loaded a separate region-per-tenant source and reported a "hard-zero cross-tenant
  leak across declared edges (unmaterialisable)." That zero is a STRUCTURAL artifact
  of the loader's Python pre-filter (each region source only ever contained that
  region's rows) — it survives removing `with_tenant` entirely, so it demonstrated
  nothing about the engine. Reframed (verified by-design on 0.26.2) to the two
  genuine isolation layers + the honest caveat:
  - **Catalog-listing isolation (hard zero):** `list_sources` filters the registry
    to `tenant_id = $cur OR IS NULL`; tenant A's listing excludes a source
    registered under tenant B (`listing_leak` = 0).
  - **Row-level discriminator-column isolation (hard zero):** the analyzer injects
    `tenant_id = $cur OR IS NULL` onto a `TableScan` *only* when the table carries a
    `tenant_id` column, so one tagged source returns disjoint rows under A vs B
    (`discriminator_leak` = 0; A reads its 989 NA rows, B its 605 EU rows).
  - **The caveat (positive assertion):** a discriminator-less source is GLOBALLY
    readable — tenant A reads ALL of B's rows when it names the source
    (`global_source_visible` = 605, the full EU set). The engine does not
    authenticate; access-gating lives above it (a discriminator column, or a Flight
    SQL / gRPC interceptor).
- **The `with_tenant` reality, stated accurately.** `with_tenant` binds the scope
  to the connection *in place* and returns `None` (the `_api_reference.md` entry and
  `rails.tenant` docstring corrected from the prior `-> Database`); it drives the
  two layers above but does NOT make a discriminator-less source's data invisible.
  `rails.tenant` keeps the bind-in-place + restore-prior-scope context manager.
- **Tests made genuine.** The prior two-tenant test passed with zero tenant scoping
  (the Python pre-filter alone). Replaced with three tests that each fail if the
  property broke: catalog-listing isolation (A's `list_sources` excludes B's
  source), discriminator-column row isolation (one tagged source, disjoint rows
  under A vs B), and the caveat (a discriminator-less B-registered source read in
  full under A). Plus negative tests for both `assert_*_isolated` helpers.
- **Gate (all green):** `ruff check` · `check_api_reference.py` (0.26.2) ·
  `no_deferral_grep.sh` · `pytest` · `quarto render` end-to-end on CPU against the
  committed cache; checksums verify; the tenancy air cells execute live (embedded
  engine — the catalog/SQL isolation behaviour is transport-independent).

### `k-rails`
- **The three rails made first-class + the closed eval loop.** Built on the merged
  arxiv + air caches; no new emit. Two chapters added (`05-rails`, `06-closed-loop`)
  wired into a new book part.
- **Provenance rail.** `rails.provenance(result)` extracts the `source` fact +
  `context_ref` off a `predict_with_context_predictor`-shaped result. The chapter
  shows a tier-04 prediction's *exact* informing rows reconstructed from the
  committed cache: the tier-04 context was assembled over the declared citation
  edges (`source = "edges"`), so a target paper's `context_ref` is its in-pool
  citation neighbourhood — read straight from `arxiv.cite_edges` + `arxiv.papers`,
  every member asserted to be a real row in the committed pool, the same-subject
  share consistent with the frozen homophily. Honest note recorded: the per-row
  `source`/`context_ref` are **not** committed as a parquet column (and
  `predict_with_context_predictor` / `assemble_context` cannot run on CPU — they
  need a source-bound GPU-emitted embedding table, and there is no
  attach-embedding-table surface on the embedded `Database`), so the trail is
  reconstructed from the committed declared-edge graph rather than replayed from a
  live predictor call. This is the auditable provenance, computed from committed
  artifacts.
- **Tenancy rail.** Reuses the corrected KV-air helpers verbatim
  (`assert_listing_isolated` / `assert_rows_isolated`); the chapter states the true
  two-layer model + caveat and asserts the frozen `air.tenancy` record (listing leak
  0, discriminator leak 0, global-source visible 605). No false "separate source
  hides data" claim reintroduced.
- **Measurement rail + the honest surface gap.** Every recipe ends in a
  `rails.measure` verdict. The chapter flagged the R1 gap as it then stood:
  `eval_embeddings` / `eval_compare` / `eval_inference` were **not yet on
  `RemoteDatabase`** — they have since landed on the wire (engine PRs #171/#172) and
  are validated cross-transport in **chapter 14**. The keystone's recall is still
  measured by its exact cosine-kNN numpy fold over the committed embedding matrix and
  asserted via `rails.measure` (recomputed 0.538 within tolerance of the golden) —
  because its committed vectors are a bare matrix, not a source-bound engine
  embedding table, so `eval_embeddings` could not score them even on the wire (see
  chapter 14's "validates the wire surface, not the arxiv keystone" framing).
  `eval_calibration` **is** on the wire (R2) and is named as the tier-04 calibration
  surface.
- **Closed eval loop.** `06-closed-loop` runs the whole spine as one measured pass
  and asserts the full golden chain: recall **0.538 → 0.556 → 0.548**; the gaussian
  predictor fits (cal-mean **2018.55** / test-mean **2018.64**, real spread); both
  conformal cruxes UNDER-cover, recomputed **live** on CPU (engine
  `conformalize_interval` → **0.830**; engine APS `conformalize` → **0.867** — both
  reproduce the frozen goldens under real execution); weighting restores neither
  (regression exact no-op Δ = 0.0; classification kNN largest mover, no scheme
  reaches nominal). The live recompute matching the goldens is the proof the chain
  holds under execution, not just as transcribed constants.
- **Tests.** `tests/test_closed_loop.py` (5 cache-backed tests): the recall chain
  reads in order; marginal-classification and regression-interval conformal each
  recompute *live* to the golden via `rails.measure`; weighting restores neither
  crux; the provenance rail reconstructs the informing rows from the cache.
- **Gate (all green):** `ruff check` · `check_api_reference.py` (26 surfaces, 0.26.2)
  · `no_deferral_grep.sh` (28 files) · `pytest` (**42 pass**) · `quarto render`
  end-to-end — the rails chapter (13 cells) and closed-loop chapter (15 cells)
  execute live against the committed cache, every golden assertion passing under
  real execution; the rendered HTML carries the real output ("full golden chain
  asserted", marginal 0.867, provenance source=edges).

### `k-bridge`
- **Four signature chapters + the Neptune-contrast framing + the verified citation
  map**, built on `k-rails`; reads the cache, no emit. One chapter
  (`07-bridge/bridge.qmd`) wired into a new book part.
- **Signature chapters (demonstrated, not asserted).** (1) Propagation = low-pass
  graph filter = SGC/APPNP = vector-agg — the `weighting` knob *is* the filter
  choice; measured on `arxiv.propagated` (recall 0.538 → 0.556). (2) An edge is a
  self-`search`; a context set is a `search`/walk — `build_neighbor_graph` /
  `assemble_context` as one retrieval primitive; the self-kNN edge carries its
  similarity+rank. (3) Graph-supervised metric learning = contrastive fine-tune over
  declared-edge walks — `fine_tune_graph(declared)`, the circularity contract shown
  on the tier-03 recall gain (declared beats base). (4) A prediction is a
  context-conditioned posterior + the conformal-under-shift crux (the moat) —
  measured on the tier-04 goldens (predictor fits ≈2018.6; both cruxes under-cover).
- **Citation verification — the rigor-critical part, done by independent research,
  NOT trusting the spec's table.** Every author/year/venue confirmed against the
  primary source; each backed by a `references.bib` entry. Three corrections the
  spec's representative table got loose on:
  - **APPNP authorship.** Spec said "Klicpera 2019". The author Johannes Klicpera
    **renamed to Johannes Gasteiger**; canonical attribution is **Gasteiger,
    Bojchevski & Günnemann, ICLR 2019** ("Predict then Propagate", arXiv 2018).
    Bib cites Gasteiger with the name-change noted.
  - **Barber et al. 2023 is a journal paper.** "Conformal prediction beyond
    exchangeability" is **Annals of Statistics 51(2):816–845, 2023** (Barber,
    Candès, Ramdas, Tibshirani) — pinned to the journal, not a preprint.
  - **arXiv-year vs formal-venue.** Angelopoulos & Bates (arXiv 2021) → *Foundations
    and Trends in ML* 16(4), **2023**; TabPFN (Hollmann et al., arXiv 2022) →
    **ICLR 2023**. Cited by arXiv year (matching the spec) with the formal venue in
    the bib `note`.
  - The rest held exactly: SGC (Wu, ICML 2019), node2vec (Grover & Leskovec, KDD
    2016), GraphSAGE (Hamilton, NeurIPS 2017), CNP (Garnelo, ICML 2018), ANP (Kim,
    ICLR 2019), Vovk (Springer 2005), Tibshirani (NeurIPS 2019), CF-GNN (Huang,
    NeurIPS 2023), NAPS (Clarkson, ICML 2023), Stanković Parts I/II/III (arXiv
    1907.03467 / 1909.10325 / 2001.00426; Parts I/II are 2019, Part III is 2020 —
    cited as the "2020" series).
- **The map is data + two enforced contracts.** `jammi_cookbook/citation_map.py`
  holds the map; the rendered table renders *from* that data (rendered == asserted).
  `tests/test_citation_map.py` asserts (a) every `bib_key` resolves to a
  `references.bib` entry, and (b) every row's `jammi_call` is a verb that exists in
  the grounded API reference (the map cannot cite a non-existent verb). The bridge
  chapter re-checks both inline, so the *render itself* fails on a dangling cite or
  a non-existent verb.
- **Quarto fails on a dangling `@cite`.** Pandoc only *warns* (exits 0) on an
  unresolved citation, so added `scripts/check_citations.py` — a hard gate that
  collects every `@key` used in the chapters and fails (exit 1) on any not in
  `references.bib`. Wired into CI before `quarto render`; negative-tested (a bogus
  key fails, all 16 real keys resolve).
- **Gate (all green):** `ruff check` · `check_api_reference.py` (26 surfaces, 0.26.2)
  · `no_deferral_grep.sh` (30 files) · `check_citations.py` (16 keys, all resolve) ·
  `pytest` (**46 pass**, incl. 4 new citation-map tests) · `quarto render`
  end-to-end — the bridge chapter (11 cells) executes against the committed cache,
  every signature's measured number asserted, all 16 citations resolved to
  reference anchors in the rendered HTML.

### `v-finetune`
- **The fine-tuning-methods vertical (A1)**: several `fine_tune` losses/knobs +
  `fine_tune_graph` measured side-by-side on the SAME task — same-subject recall@10
  over the committed ogbn-arxiv subset (the keystone's embedding-independent target,
  the exact cosine-kNN numpy fold over each method's committed embedding matrix). One
  chapter (`08-finetune-methods/finetune-methods.qmd`) in a new book part. Pinned
  `jammi_ai==0.26.2`; GPU emit on the A10G against `jammi-server-cu12==0.26.2`.
- **The shared supervision (apples-to-apples).** Plain `fine_tune` reads a supervised
  schema from the source columns; the keystone mines all three from the SAME signal —
  the subset's subject labels: `(anchor, positive)` pairs (MNRL/Matryoshka),
  `(anchor, positive, negative)` triplets (triplet loss), `(text_a, text_b, score)`
  contrastive (CoSENT). Same label signal the recall target scores — a gain is a real
  metric-learning gain, not a circular one (the tier-03 circularity contract, from the
  graph to the loss).
- **The REAL per-method recall@10** (frozen base 0.538): cosent 0.540 (Δ +0.002),
  mnrl_t0.05 0.5405 (+0.0025), mnrl_t0.20 0.542 (+0.004), triplet 0.5395 (+0.0015),
  matryoshka 0.540 (+0.002), graph_declared **0.5485** (+0.0105). best graph_declared,
  worst triplet, **spread 0.009** — a narrow band. Reproduced bit-for-bit across three
  emit runs (deterministic). **The honest finding:** on a same-subject signal the base
  ModernBERT geometry already separates, the contrastive loss / temperature / Matryoshka
  knobs move recall only within a tolerance-width band; the only method that opens a
  real (still small) gap is `fine_tune_graph`, because it trains on a *different*
  supervision (declared citation edges) the embeddings cannot reconstruct. The
  supervision caps the gain, not the loss. No method is claimed "best" beyond the literal
  number.
- **Matryoshka curve** (recall vs truncated dim, recomputed live on CPU off the committed
  matrix): dim 768 → 0.540, 256 → 0.5225, 64 → 0.4645. Monotone non-increasing; the
  64-dim prefix (12× smaller index) retains 86% of full recall — a free axis, not a free
  lunch (it does not raise full-dim recall).
- **Hard-negatives — a REAL engine-surface finding, recorded not faked.**
  `mine_hard_negatives=True` OOMs on the A10G (23 GB) in the corpus-encode pass at the
  cookbook's full supervised scale (1500 pairs), failing in the ModernBERT forward pass
  *before training starts* and *independent of `batch_size`* (probed: OOM at ≥500 pairs,
  completes at ≤300). The keystone records this as `oom_at_full_scale` with the measured
  threshold + a 300-pair run that proves the kwarg works (recall 0.537) — never a
  fabricated recall. Hard-negatives is therefore reported as a finding, kept OUT of the
  apples-to-apples recall table. Also surfaced: `mine_hard_negatives=True` requires
  `hard_negative_refresh_every > 0` (errors otherwise though the signature defaults it to
  `None`).
- **Gate (all green):** `ruff check` · `check_api_reference.py` (26 surfaces, 0.26.2; the
  `fine_tune` entry now pins `embedding_loss`/`mnrl_temperature`/`mine_hard_negatives`/
  `hard_negative_*`/`matryoshka_dims`) · `no_deferral_grep.sh` · `check_citations.py`
  (16 keys) · `pytest` (**51 pass**, +5 finetune cache tests) · `quarto render`
  end-to-end — the finetune-methods chapter (10 cells) executes live against the committed
  cache, every per-method recall asserted to golden, the Matryoshka curve recomputed live,
  the hard-neg OOM finding asserted; checksums verified.

### `chore/repin-0.26.4-tenant-migration` (#11)
- **Re-pin `jammi_ai==0.26.4` + `with_tenant` → `set_tenant` / `tenant_scope`
  migration.** Updated `pyproject.toml`, chapters, `_api_reference.md`, and
  `check_api_reference.py` to pin 0.26.4. All `with_tenant(t)` call sites migrated to
  `set_tenant(t)` (sticky setter) or `with db.tenant_scope("t"):` (block-scoped CM)
  depending on usage; `rails.py` updated to use `tenant_scope` internally. The API
  reference was re-grounded against the 0.26.4 wheel: `set_tenant` / `tenant_scope` /
  `tenant` replace `with_tenant`, and the eight cp9 verbs (`create_mutable_table`,
  `drop_mutable_table`, `list_mutable_tables`, `register_topic`, `drop_topic`,
  `list_topics`, `publish_topic`, `subscribe_collect`) were added to
  `check_api_reference.py` and documented in `_api_reference.md` (commit `0921cc4`).
  The guard now checks 36 surfaces: 34 REQUIRED `Database` verbs + 2 module
  functions (`open_local`, `connect`).
- **Gate (all green — the CI `book` check passed on the merged PR):** `ruff check` ·
  `check_api_reference.py` (36 surfaces, 0.26.4) · `no_deferral_grep.sh` ·
  `check_citations.py` · `pytest` · `quarto render` end-to-end.

### `feat/c1-mutable-feature-store` (C1, #12)
- **Chapter 12 — mutable companion table as a feature store.** Feature: each paper's
  citation in-degree over the committed declared citation graph — graph-derived and
  deterministic, committed by `scripts/build_feature_store_cache.py` as
  `artifacts/feature_store/paper_features.parquet`. The chapter provisions a mutable
  table with a schema and primary key via `create_mutable_table`, inserts the feature
  rows via `sql(INSERT INTO mutable.public.<name> …)`, and JOINs the feature into a
  query over `arxiv.papers` — the federating pattern. The `SUM(in_degree)` aggregate
  over the committed citation sub-graph is asserted against the frozen golden.
  `list_mutable_tables` is called and its output asserted. The chapter runs on CPU with
  the embedded engine (transport-independent behavior; the `create_mutable_table` /
  `list_mutable_tables` verb surface is the same on `RemoteDatabase` by the 0.26.4
  conformance guard).
- **API guard:** the eight cp9 verbs (`create_mutable_table`, `drop_mutable_table`,
  `list_mutable_tables`, `register_topic`, `drop_topic`, `list_topics`,
  `publish_topic`, `subscribe_collect`) were already in `check_api_reference.py` and
  `_api_reference.md` — added by #11 (commit `0921cc4`). #12 verified they were
  present and added no new guard entries (per its merge commit: "verified, not
  duplicated").
- **Gate (all green — the CI `book` check passed on the merged PR):** `ruff check` ·
  `check_api_reference.py` (0.26.4) · `no_deferral_grep.sh` · `check_citations.py` ·
  `pytest` (incl. `test_feature_store_cache.py`) · `quarto render` end-to-end — the
  feature-store chapter executes live against the committed cache, the JOIN
  aggregate asserted to golden.

### `feat/c2-cdc-triggers` (C2, #13)
- **Chapter 13 — CDC with trigger topics.** A CDC stream of 60 record-change events
  (op = `add` / `update` / `remove`; key = committed arxiv `paper_id`; offsets 0–59)
  published onto a registered topic; the chapter demonstrates `subscribe_collect` in
  three modes: bounded replay from offset 0, predicate-filtered subscribe
  (`op = 'add'`), and replay from the mid-stream checkpoint at offset 24. Commit
  offsets, row counts, and the filtered add-count are asserted against the frozen
  golden (`artifacts/cdc/golden_metrics.json`: `replay_count` 60, `add_count` 20,
  `tail_count` 36 from offset 24, `tail_add_count` 12 — all zero-tolerance). The
  chapter runs hermetically on CPU with the embedded engine using the in-memory
  broker (no NATS, no server). `scripts/build_cdc_cache.py` emits the golden;
  `tests/test_cdc_cache.py` asserts the counts.
- **New cite:** `@kreps2011kafka` — Kreps, Narkhede & Rao, "Kafka: a Distributed
  Messaging System for Log Processing", NetDB Workshop 2011 — independently verified
  and added to `references.bib`.
- **Gate (all green — the CI `book` check passed on the merged PR):** `ruff check` ·
  `check_api_reference.py` (0.26.4) · `no_deferral_grep.sh` · `check_citations.py` ·
  `pytest` (incl. `test_cdc_cache.py`) · `quarto render` end-to-end — the CDC chapter
  executes live, every count asserted to golden.

### `feat/ch14-eval-channels` (#15)
- _(Audit entry back-filled at the H3 Step-0 reconcile, from the merge commit + the
  committed cache; the per-spec table row above carries the full measured detail.)_
- **Chapter 14 — eval on the wire + provenance channels.** The engine↔cookbook
  validator for the eval verb family (`eval_embeddings` / `eval_per_query` /
  `eval_inference` [classification + NER] / `eval_compare`) and the channel registry
  (`register_channel` / `add_channel_columns` / `list_channels`) now on
  `RemoteDatabase` (engine #171/#172). `scripts/build_eval_cache.py` runs every verb +
  the channel sequence on BOTH transports (embedded in-process + a live `grpc://`
  `jammi-server`) over the engine's public deterministic fixtures (20-row patents
  corpus, tiny_modernbert/tiny_bert encoders, classifier + NER models) and asserts
  `remote == embedded` live — shape equality + value-closeness at 1e-9, NER
  `Entity.confidence` at its true f32-relative 1e-6 (the engine computes that span score
  in f32). The parity verdict is recorded once in `artifacts/eval/eval.json`; PR CI
  reads the committed embedded-canonical cache on CPU (aggregates-to-golden + report
  shape) and runs the channel half live-embedded (re-deriving the #170
  tenant-isolation / non-collision property each render).
- **Honest framing (in prose, not tuned away):** this validates the wire SURFACE as a
  golden-stability gate on the engine's public fixtures; it does **not** extend cookbook
  corpus coverage and does **not** reproduce the keystone 0.538/0.556 recall (those stay
  the ch05 numpy fold, forced by the keystone's bare-matrix arxiv vectors).
- **Measured (embedded canonical):** embeddings recall@k 0.2833 / mrr 0.2722 / ndcg
  0.2262; classification acc 0.4 / f1 0.2857; NER p/r/f1 0.0 (random-init fixture —
  well-defined rates in [0,1], wire runs end-to-end); `eval_compare` self-anchor
  max\|Δ\| 0.0; two-table (tiny_modernbert vs tiny_bert) recall Δ +0.2333 (p 0.2858).
- **Surfaced upstream before release:** engine #173 (`eval_compare` significance CIs
  made order-invariant — `bootstrap_ci` canonicalizes before resampling) and the
  NER-confidence f32-precision question. Pinned `jammi_ai==0.26.5`.
- **Gate (all green — CI `book` passed on the merged PR):** `ruff check` ·
  `check_api_reference.py` (0.26.5) · `no_deferral_grep.sh` · `check_citations.py` ·
  `pytest` (incl. `test_eval_cache.py` + `test_channels.py`) · `quarto render`.

### `feat/scale-tier-arxiv-emit` (#16)
- _(Audit entry back-filled at the H3 Step-0 reconcile, from the merge commit.)_
- **The H2 W1 scale tier — 170k arxiv ANN-vs-exact recall.** The full
  169 343-paper ogbn-arxiv graph embedded once on the GPU box with ModernBERT
  (title+abstract, 768-dim), partitioned into a 168 343-vector indexed corpus + a
  **held-out** 1 000-vector query set (the last 1 000 `paper_id` by ascending sort,
  disjoint by construction). The frozen USearch sidecar is built once by the engine's
  own `SidecarIndex` builder (USearch 2.25.1). The recall floor is **meaningful** — each
  held-out query's true nearest neighbour is a different paper, not a structural
  recall@1≈1: measured recall@1 0.971 / recall@10 0.956 / recall@100 0.915, committed as
  floors (measured − 0.04 margin). ANN QPS (1115 @k=100 on the A10G) recorded un-gated as
  a perf reference.
- **Commit-once-read-forever.** Large artifacts (`arxiv_vectors.parquet`,
  `scale_query_vectors.parquet`, `arxiv_ann.usearch`, `arxiv_ann.rowmap`) in Git LFS;
  JSON/txt sidecars plain text; CI pulls LFS and never recomputes. `build_scale_cache.py`
  is a new sibling of the 4k keystone `build_arxiv_cache.py` (does not mutate it or
  `data/ids/arxiv.txt`). The `scale.qmd` chapter checksum-verifies the cache, builds an
  exact numpy cosine-kNN oracle over the full corpus, loads (never rebuilds) the frozen
  index, and asserts recomputed recall@{1,10,100} against the committed floors at render.
- **Gate (all green — CI `book` passed on the merged PR):** `ruff check` ·
  `check_api_reference.py` · `no_deferral_grep.sh` · `check_citations.py` · `pytest` ·
  `quarto render` end-to-end against the LFS scale cache.

### `feat/scale-recall-vs-cost` (#17)
- _(Audit entry back-filled at the H3 Step-0 reconcile, from the merge commit.)_
- **Chapter 14-scale extended to the recall-vs-COST curve + the usearch-drift close.**
  The HNSW knob trade a practitioner actually tunes against an SLA, on two lifecycles:
  - **`ef_search` (query-time), GATED.** Re-dial `index.expansion_search` over the SAME
    committed frozen index (no rebuild) and recompute recall@10 vs the same numpy oracle,
    measuring QPS: recall climbs 0.689 → 0.996 across ef 8→256 while QPS falls; each point
    clears a committed per-ef floor — a genuine render-time re-derivation, a real gate.
    The ef=64 point reproduces the engine's committed recall@10 0.9564.
  - **connectivity / `build_expansion` (build-time), UN-GATED reference.** Each setting is
    a separately built ~542 MiB graph (LFS-prohibitive to commit per point); the engine's
    `jammi-bench recall-sweep` builds them once on-box and only the measured cost curve
    (`recall_sweep_build.json`) is committed: build time 144 → 424 s with `build_expansion`,
    index size grows with connectivity, recall holds. Not re-derivable in CI by design —
    the build-cost columns are the deliverable, recall is provenance.
- **Usearch version-drift hole closed.** The frozen index was built with usearch 2.25.1
  but the dep floated (`>=2.25.1,<3`) so CI resolved 2.25.3, and the file header carries
  only the major version — a silent cross-version load whose recall the manifest only
  *printed*, never *asserted*. Pinned usearch **exactly 2.25.1** and assert
  `usearch.__version__ == manifest` before any recompute, so future drift fails loudly.
  Cross-check: the engine's own `jammi-bench search_sweep` (fresh index) agrees with the
  chapter's Python re-dial (committed index) on recall@10 within 1e-3 across the grid.
- **Gate (all green — CI `book` passed on the merged PR):** `ruff check` ·
  `check_api_reference.py` · `no_deferral_grep.sh` · `check_citations.py` · `pytest` ·
  `quarto render`.

### `chore/reconcile-0.29.0-upgrade` (#18)
- **Step-0 reconcile + `jammi_ai` re-pin `0.26.5 → 0.29.0`, ahead of the regression
  chapter.** `pyproject.toml` pin bumped (the 0.29.0 line adds the scale tier + the
  scale-robust regression fine-tune surface); `_api_reference.md` **re-introspected
  against the installed 0.29.0 wheel** — `fine_tune`'s regression surface added (`seed`,
  `regression_loss`, `regression_beta`, `quantile_levels`) with the precise beta-NLL
  framing (stop-gradient re-weighting that prevents mean-starvation, distinct from the
  std floor); EXECUTION-STATUS flipped #15 → merged and added #16/#17.
- **Gate (all green — CI `book` passed on the merged PR):** `ruff check` ·
  `check_api_reference.py` (**40 surfaces**, 0.29.0) · `no_deferral_grep.sh` ·
  `check_citations.py` (32) · `pytest` (**92 pass** — no conformal/search golden drift
  across the six-version jump) · `quarto render` (**19 chapters**).

### `feat/finetune-regression-chapter` (#19)
- **Chapter 15 — regression fine-tuning on a high-offset target (the H2 W5 loop-closer).**
  The public `fine_tune(task="regression")` surface measured: predict ogbn-arxiv `year`
  (~2018, high-offset) from title+abstract across all four `regression_loss` objectives
  (`beta_nll` default / `gaussian_nll` / `crps` / `pinball`), inferred on a **seeded
  representative held-out split**, folded to held-out RMSE-in-years + nominal coverage
  from the committed **de-standardized** predictions. The committed held-out predictions
  let the chapter/test **re-fold** every metric — auditable, never hand-written.
- **Honest measured findings (recorded, not tuned away):**
  - the high-offset target **FITS WITHOUT COLLAPSE** — Gaussian heads serve a real spread
    (min `std_mean` 0.75y) vs the documented pre-0.26.2 `std≈0.001` degenerate failure.
    **This is the v0.29.0 z-space-loss win, measured.**
  - `year` is **low-signal from text** — every objective regresses to the conditional
    mean (RMSE ≈ test std ≈ 1.05y), so the loss choice is a near-tie (crps marginally
    best); no objective is claimed "best" beyond the literal number.
  - Gaussian intervals are mildly **overconfident** (coverage 0.736 < 0.90); the pinball
    quantile band **collapses** (width 0.215y → 0.000 coverage) — a candidate engine
    observation, recorded honestly.
- **Artifacts.** GPU emit `scripts/build_finetune_regression_cache.py` (0.29.0-guarded,
  seed-recorded) + `tests/test_finetune_regression_cache.py` + the 5-artifact contracts
  registry + `references.bib` (`nix1994` / `seitzer2022` / `koenker1978`, independently
  verified).
- **Gate (all green — CI `book` passed on the merged PR):** `ruff check` ·
  `check_api_reference.py` (**40**, 0.29.0) · `no_deferral_grep.sh` ·
  `check_citations.py` (**35**) · `pytest` (**97 pass**) · `quarto render`
  (**20 chapters**).
