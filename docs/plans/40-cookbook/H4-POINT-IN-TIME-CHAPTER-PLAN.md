# H4 — Point-in-time correctness: the leakage-free training set

> Chapter plan, authored in the cookbook's wave-plan style (cf. `H3-WAVE-PLAN.md`). One PR. Adds one chapter, one build script, one test, one golden-metrics artifact, and the bib entries it cites. Tracking row goes in `EXECUTION-STATUS.md` before implementation starts.
>
> **Status:** spec — draft (hand-off). **Gated** on the engine shipping the temporal-correctness primitives (see Gate). Until then this is a plan, not a buildable chapter.

## Why this chapter

Chapter 12 (`12-feature-store/feature-store.qmd`) demonstrates the *easy half* of a feature store: a mutable companion table federated into a query by key. It is honest about its limit — it federates **current-state** features, with no notion of "the value as it was known at time T." That is exactly the half a plain `JOIN` already does well, and the half that leaks the future into the past the moment a label carries a timestamp.

This chapter demonstrates the **hard half**, the half that makes a feature store a feature store rather than a join: **point-in-time correctness**. It is built on the engine's `asof_join` and `verify_materialization` primitives (jammi-ai `docs/plans/60-temporal-correctness/`). The measured lesson is sharp and uncomfortable: assembling a labelled set with a current-state join silently inflates a downstream metric *and* inflates conformal coverage, while the as-of join keeps both honest. The chapter ends in real numbers that show the leak and show it closed.

This is the cookbook half of the engine↔cookbook evolution: the chapter is the forcing function that proves the new primitive composes into a real workload. It proves the **primitive**; it does **not** build the closure against a live online serving tier — that, with a co-deployed KV store and an auth interceptor, is a production online-serving concern (a low-latency serving tier behind auth with a live coverage SLA), out of scope for this engine cookbook. Stated upfront and honestly, per house discipline.

## Gate (do not claim before this holds)

1. **Engine temporal-correctness primitives shipped and published.** jammi-ai SPEC-01 (`asof_join`) and SPEC-02 (`verify_materialization`) merged and released on a public wheel (target ≥ 0.31.x). Both verbs present on `Database` *and* `RemoteDatabase` with the conformance guard green.
2. **Pin advanced + re-introspected.** `pyproject.toml` `jammi_ai==0.31.x` (sole pin location), then `python scripts/check_api_reference.py` regenerates `jammi_cookbook/_api_reference.md` to include `asof_join` / `verify_materialization`. The chapter teaches the published surface, never a local build.
3. **Keystone cache present.** The ogbn-arxiv keystone artifacts (`artifacts/arxiv/`) are committed (already true post-H1).

If (1) slips, this chapter stays a plan. No stubbing the verb, no "for now" — the no-deferral grep (`scripts/no_deferral_grep.sh`) forbids it.

## Placement

New part-member under **"Streaming & mutable state,"** sibling to Ch 12 (feature-store) and Ch 13 (CDC):

- Directory: `chapters/19-point-in-time/point-in-time.qmd`
- `_quarto.yml`: add after `13-cdc/cdc.qmd` in the streaming part.

## Header block (the Recipe / Theory / Rail contract)

```
---
title: "Point-in-time correctness — the feature value as it was known, not as it is now"
---

```{python}
# | echo: false
import jammi_cookbook
```

**Recipe:** `asof_join` (backward · inclusive · `by` entity · `tolerance` look-back ·
deterministic tie-break) · `verify_materialization` (the materialized training set's
definition hash + input as-of anchors) · the contrast against a naive current-state
`JOIN`. · **Theory:** point-in-time correctness and label leakage [@kaufman2012leakage],
bitemporal valid-time vs transaction-time [@snodgrass1999temporal], the feature store's
reason to exist is the as-of join [@orr2021featurestore], and why a leaky split breaks the
exchangeability conformal coverage rests on [@barber2023beyond]. · **Rail:** measurement
(the leakage delta and the train/serve-skew, each asserted against the frozen golden — skew
exactly zero, leak strictly positive).
```

## The measured argument (section ordering)

Follows the house skeleton: state the recipe and the honest limit upfront, then the implementation, then the measured assertion that ends in real numbers.

1. **The leak, stated plainly.** A label on paper *p* observed at time *T* (e.g. "was *p* cited ≥ k times within its first year"). A feature `citation_in_degree(p)` computed over the *committed* declared citation graph. The naive recipe joins the label spine to the **current** in-degree — which counts citations that arrived *after* T. That is leakage: the feature knows the future of the label.

2. **The as-of fix.** Build a time-stamped fact relation: each citation edge carries the `cited_at` of the citing paper (committed, graph-derived, in `artifacts/point_in_time/`). `asof_join` the label spine (`paper_id`, `as_of = T`) to the per-edge facts, `by = paper_id`, `time = cited_at`, `direction = Backward`, `boundary = Inclusive`, counting only edges at-or-before T. One verb; the leakage-safe in-degree.

3. **Leakage delta — measured.** Compute the downstream metric (a simple linear-probe AUC, or the recall of a "highly-cited" classifier) two ways: trained on the naive current-state features vs the as-of features, both evaluated on a held-out as-of-correct test split. The naive number is **optimistically inflated**. Golden: `pit.leakage_delta = naive_auc − asof_auc` is strictly positive and pinned to tolerance; `pit.asof_auc` pinned.

4. **The conformal honesty result — the killer measurement.** Calibrate a split-conformal predictor (`conformalize*`) two ways: on a leaky current-state calibration split vs an as-of-correct one. Measure empirical coverage on the as-of-correct test set. The leaky calibration **over-covers in appearance** (its scores look easier than deployment), so its *claimed* coverage overstates reality; the as-of calibration tracks nominal. Golden: `pit.coverage_leaky` (inflated, > nominal+band) vs `pit.coverage_asof` (≈ nominal, tol 0.02). This is [@barber2023beyond] made concrete: leakage is a non-exchangeability that silently breaks the guarantee.

5. **Train == serve, by construction.** Run the *same* `asof_join` definition twice — embedded (`jammi_ai.connect("file://…")`, the offline training-set path) and via `grpc://` (the serving-read path) — over the same committed inputs, and assert the feature vectors are identical. Golden: `pit.train_serve_skew = 0.0` (exact). The point: one definition, both paths; skew collapses because the definition is shared, not re-implemented. (Dual-transport pattern, cf. Ch 14/16/18.)

6. **The artifact knows what it is.** `verify_materialization` on the materialized as-of training table returns `Match` against its own `definition_hash`; re-deriving with a changed query yields a different hash (a stale copy is detectable); an input with no version surface surfaces `MatchWithUnpinnedInputs` honestly. Golden: `pit.definition_hash` committed (exact); the mismatch case asserted as a flow event.

7. **The honest limit (closing).** This chapter proves the primitive composes into a leakage-free, skew-free training surface — on CPU, against the committed cache. It does **not** serve those features from a low-latency online tier behind auth with a live coverage SLA; carrying the materialization contract across that boundary so the online read can assert "what I serve matches what trained this, as of T" is a production online-serving concern, not something this engine cookbook covers. One sentence, pointing forward; no overclaim.

## Determinism, cache, and CI (non-negotiables)

- `import jammi_cookbook` first (pins OMP/MKL/OpenBLAS threads, tokenizer parallelism, seed).
- `build_neighbor_graph(exact=True)`; subset identity committed under `data/ids/` (the as-of label spine and the time-stamped edge facts are committed, not seeded).
- **Read the cache, never recompute upstream.** Heavy embedding/graph work is the keystone's; this chapter loads `arxiv.*` artifacts and its own `point_in_time.*` facts, and asserts against `artifacts/point_in_time/golden_metrics.json`.
- Every measured cell ends in `rails.measure(key, value)` followed by an assertion against the committed golden via `contracts.assert_close(key, value)` (tolerances per metric; `pit.train_serve_skew` and `pit.definition_hash` are `tol: 0` / exact).
- No-deferral grep and citation check must pass; `quarto render` executes the chapter in CI and validates every metric.

## Build script, test, artifacts

- `scripts/build_point_in_time_cache.py` — CPU, hermetic, ephemeral `file://` catalog. Builds: the time-stamped citation-edge fact table (`cited_at` per edge), the as-of label spine, the two training-set materializations (naive vs as-of), the downstream-metric and coverage numbers, and the `definition_hash`. Emits `artifacts/point_in_time/{point_in_time.parquet, golden_metrics.json, checksums.json}`. CLI `--target` for an optional `grpc://` arm (the skew check); defaults to embedded.
- `tests/test_point_in_time_cache.py` — loads the artifacts, sanity-checks shapes, asserts every golden within tolerance, and asserts `train_serve_skew == 0` and the `definition_hash` round-trip.
- `artifacts/point_in_time/golden_metrics.json` keys: `pit.leakage_delta`, `pit.naive_auc`, `pit.asof_auc`, `pit.coverage_leaky`, `pit.coverage_asof`, `pit.train_serve_skew` (tol 0), `pit.definition_hash` (exact).

## Citations to add to `references.bib`

Each verified against the primary source before commit (per the bib header discipline):

- `@kaufman2012leakage` — Kaufman, Rosset, Perlich, Stitelman, "Leakage in Data Mining: Formulation, Detection, and Avoidance," ACM TKDD 2012. https://dl.acm.org/doi/10.1145/2382577.2382579
- `@snodgrass1999temporal` — Snodgrass, *Developing Time-Oriented Database Applications in SQL*, Morgan Kaufmann, 1999 (valid-time vs transaction-time).
- `@barber2023beyond` — Barber, Candès, Ramdas, Tibshirani, "Conformal prediction beyond exchangeability," *Annals of Statistics* 51(2), 2023. https://arxiv.org/abs/2202.13415
- `@orr2021featurestore` — already cited by Ch 12; reuse.
- `@kleppmann2017ddia` — already present; reuse for the log-vs-state framing if needed.

## `EXECUTION-STATUS.md` row (add before implementation)

```
| H4-PIT point-in-time / leakage-free training set | open | `feat/h4-point-in-time` | — | chapter 19: `asof_join` (backward/inclusive/by-entity/tolerance) + `verify_materialization`; time-stamped citation-edge facts (graph-derived, committed parquet); measures leakage delta (naive vs as-of), conformal coverage honesty (leaky vs as-of calibration), train/serve skew = 0 across embedded==grpc, and the materialization definition-hash round-trip; `scripts/build_point_in_time_cache.py` + `tests/test_point_in_time_cache.py`; pin → jammi_ai==0.31.x. Closure against a live online tier is a production online-serving concern, out of scope here. |
```

## Discipline check

Names no consumer: the chapter uses ogbn-arxiv (public), and the word "feature store" appears only as the *theory* the recipe illustrates, never as an engine concept — exactly as Ch 12 already does. The recipe is `asof_join` + `verify_materialization`, both general engine primitives; a reader who has never heard the words "feature store" still wants leakage-free temporal joins. References point one way: cookbook → `jammi_ai`. If the engine surface is missing or wrong, escalate to the engine repo (`60-temporal-correctness`), never patch around it here.
