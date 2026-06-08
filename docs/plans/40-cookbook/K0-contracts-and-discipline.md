# K0 — Contracts & execution discipline

**Status:** spec — draft (hand-off). Read after `README.md`, before `K1`.
**Purpose:** the load-bearing infrastructure decisions every other cookbook spec depends on — the artifact-contract registry, the author-vs-execute discipline, the determinism template, and the philosophy gates. This is *not* an M4-style parallel coordinator playbook; the cookbook is built serially (see README "Build order").

---

## 1. The artifact-cache spine (two-layer contract registry)

The cookbook's heavy work (embedding, neighbor-graph build, fine-tune, context-predictor train, calibration) runs **once**, in the KV-arxiv keystone slice. Everything else is authored **against the recorded results**. This is the single most important mechanism in the cookbook; get it right first.

A registry module (`jammi_cookbook/contracts.py`, created by K1) holds **two layers**:

### Layer 1 — Schema (enables parallel *code* authoring)

For each artifact a tier produces/consumes: a stable **name**, **columns + dtypes**, partition/key columns, and — for graph context — the exact **declared-edge params** (`edge_source`, `edge_src_column`, `edge_dst_column`, `edge_type_column`, `edge_weight_column`, `edge_hops`, `edge_fanout`, `edge_direction`, `edge_types`, `min_weight`, `hybrid`). Example entries:

```
arxiv.embeddings      : parquet (_row_id Utf8, vector FixedSizeList<f32,768>)       produced by Tier01
arxiv.neighbor_graph  : edge table (src Utf8, dst Utf8, rank Int, similarity f32)    produced by Tier01 (build_neighbor_graph)
arxiv.cite_edges      : edge table (src Utf8, dst Utf8)                               external declared edges (the citation graph)
arxiv.propagated      : parquet (_row_id, vector)                                     produced by Tier02 (propagate_embeddings)
arxiv.ft_model        : model_id (str)                                               produced by Tier03 (fine_tune_graph)
arxiv.ctx_predictor   : model_id (str)                                               produced by Tier04 (train_context_predictor)
arxiv.cal_split       : the (calibration / test) row-id partition                    produced by Tier04
```

### Layer 2 — Golden samples (enables *measured-verdict* authoring + is the CI oracle)

The committed small-subset **artifact files themselves** plus `golden_metrics.json`:

```json
{
  "arxiv": {
    "tier01.recall_at_10": {"value": 0.74, "tol": 0.02},
    "tier03.recall_at_10": {"value": 0.81, "tol": 0.02},
    "tier04.marginal_coverage": {"value": 0.83, "tol": 0.03},   // the under-coverage the crux depends on
    "tier04.repaired_coverage": {"value": 0.90, "tol": 0.02},   // restored by the inline graph-aware repair
    "tier04.mean_set_size": {"value": 2.4, "tol": 0.3}
  }
}
```

**Mandate (the chief risk control): chapters READ the cache; they do not recompute upstream.** Re-execution → numeric drift → CI tolerance failures with no provenance is the #1 hand-off failure mode. The keystone slice produces the cache once; every later chapter loads `arxiv.*` artifacts and asserts against `golden_metrics.json`.

## 2. Author vs. execute (the two phases)

Conflating these is the trap that makes "parallelism" fake and CI flaky. Keep them distinct:

- **Author** (cheap, no heavy execution): write prose, API calls, citations, bridge notes, and the *measured-verdict cells that read cached artifacts + golden metrics*. Any session can do this without a GPU.
- **Execute & measure** (expensive, one machine, **once**): the keystone slice runs the real pipeline and emits the committed artifacts + golden metrics. The integration run (and the opt-in full-scale run) re-executes end-to-end.

CI executes the notebooks (`--execute`) on the **small fixed subset**, loading the **committed checkpoint/artifacts** (it never retrains, never re-embeds the full set) and asserting metrics to tolerance.

## 3. Determinism contract (template — every vertical/recipe spec embeds this)

```
- env: OMP_NUM_THREADS=1, tokenizers parallelism off, pinned dtype (f32), pinned torch threads
- ANN: exact=True; deterministic tie-break (equal similarity → order by _row_id) so recall@k is stable
- subsetting: seeded AND the selected _row_id list is COMMITTED (do not trust seed→same-nodes across lib versions)
- datasets: pinned by version + checksum; download is checksum-gated
- fine-tune / context-predictor: CI runs the COMMITTED checkpoint; retraining is the opt-in full-scale path only
- metrics: asserted to TOLERANCES, not bit-equality (BLAS matmul order varies); commit the artifact, compare against the frozen vector
- CI runs the notebooks with `--execute` (not a lint) — a moved fixture/artifact path must fail CI, not pass a lint
```

## 4. Philosophy gates (a recipe is not done unless all hold)

- **Names no consumer.** Public datasets only (Air Routes, ogbn-arxiv); neutral framing; no AccuRisk/Lace/HCC/crosswalk vocabulary. (`philosophy.md:10-12`.)
- **Runnable + measured.** Every recipe executes in CI and ends in a real number from the committed artifacts — never a placeholder. The book *is* the closed eval loop (`construct→propagate→learn→MEASURE`).
- **Conformal doctrine respected.** Tier-04 uses the **marginal** OSS Python conformal (`conformalize*`), demonstrates it *under-covers* on the citation graph, then applies the graph-aware repair **inline in the notebook** (the consumer makes the cohort/weight choice — "choosing the cohort is governance", `conformal-prediction.md:29-30`), and points to enterprise **E8** for the productionized governed version. It does **not** expose or expect Mondrian/weighted in OSS Python.
- **Engine untouched.** No edits to jammi engine repos — the cookbook only *calls* the `jammi_ai` API. A recipe that needs an engine change is a fork to escalate, not patch.

## 5. Scope boundary

This spec set produces the **cookbook**, a consumer artifact in its own repo. It targets the `jammi_ai` API (incl. `propagate_embeddings`). It is distinct from `jammi-ai/cookbook/` (S6). See README "This is NOT the S6 cookbook".
