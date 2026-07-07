# K1 — Repo scaffold + `jammi_cookbook` shared lib

**Status:** spec — draft (hand-off). Build first (with K0). One PR.
**Produces:** the cookbook repo skeleton, the shared `jammi_cookbook` Python lib, the CI `--execute` harness, the cookbook's own `CLAUDE.md`, and the embedded grounded `jammi_ai` API reference (below) so no later spec reverse-engineers signatures.

---

## 1. Repo & build tool

A **new repo** (suggested name: `jammi-cookbook`; consumer-neutral). **Quarto** book project (`_quarto.yml`, `.qmd` chapters with executable Python cells via the jupyter engine) rendering to a browsable book and executing in CI. (Jupyter Book is an acceptable substitute; pick one, record it.) `jammi_ai` is a pinned dependency (`pyproject.toml`); the repo never vendors or edits engine source.

Layout:
```
jammi-cookbook/
  _quarto.yml
  CLAUDE.md                  # the cookbook's own standards (§4)
  pyproject.toml             # pins jammi_ai==<version>, quarto deps, numpy, pyarrow
  jammi_cookbook/            # the shared lib (§2)
  chapters/                  # 01-construct/ 02-analyze/ 03-learn/ 04-predict/ bridge/ (authored by KV-*/K-*)
  artifacts/                 # committed golden-sample cache (K0 layer 2) — small subset only
  data/ids/                  # committed seeded subset _row_id lists (K2)
  .github/workflows/ci.yml   # the --execute harness (§3)
```

## 2. `jammi_cookbook` shared lib (the only Python the chapters import beyond `jammi_ai`)

- `contracts.py` — the two-layer artifact-contract registry (K0 §1): artifact names/schemas + a `golden(metric)` accessor reading `golden_metrics.json` with tolerances. Helper `load_artifact(name)` reads from `artifacts/` (never recomputes).
- `datasets.py` — `load_air_routes()` / `load_ogbn_arxiv(subset=...)` (K2): checksum-gated download, registers sources into a `jammi_ai` db, returns the committed subset using `data/ids/`.
- `rails.py` — the three rails as thin helpers (K-rails owns the deepening): `provenance(...)` (read evidence channels / `context_ref` off a result), `tenant(db, t)` (wrap `with_tenant` + the two-tenant assertion), `measure(...)` (call the right `eval_*` and assert against `golden(...)`).
- `determinism.py` — sets the env/seed contract (K0 §3) on import; exposes `seeded(name)` and a `committed_ids(dataset)` loader.

The lib stays small: it *composes* `jammi_ai` and *enforces* the contracts/rails — it implements no graph/ML logic of its own.

## 3. CI `--execute` harness

`ci.yml` runs: install pinned deps → fetch committed artifacts (in-repo) → `quarto render --execute` (or `jupyter nbconvert --execute`) on the **small-subset config** with the determinism env → assert every notebook's measured cells match `golden_metrics.json` within tolerance → fail on any band-aid tell-sign (the no-deferral grep, README §2). The full-scale run is a separate, opt-in (manual/nightly) workflow that *may* retrain; the PR-gating workflow never does.

## 4. The cookbook's own `CLAUDE.md`

Encodes the local half of the fork-resolution harness (README §3.1):
- **Determinism** is mandatory (K0 §3); **read the cache, never recompute upstream** (K0 §1).
- **Names no consumer**; public datasets only; neutral prose.
- **Runnable + measured**: every recipe executes in CI and ends in a real number; no placeholders (the no-deferral policy, README §2).
- **Conformal doctrine**: marginal in OSS Python; graph-aware repair inline; governed version is E8 (K0 §4).
- **Don't touch engine repos.** Engine changes are forks to escalate, not patch.

## 5. `jammi` Python API reference (embed verbatim)

These are the `jammi` Python signatures the chapters call. Pin the `jammi-ai` dist version in `pyproject.toml`; call these exactly — confirm against the installed package (the bindings live in `crates/jammi-python/src/`), never invent or assume a signature.

**Setup / sources**
- `jammi.connect(target) -> Session` — a `file://<artifact_dir>` target opens the in-process embedded engine (an `EmbeddedBackend`, via the `jammi-ai[embedded]` extra); `grpc://`/`https://` opens a remote `RemoteDatabase`. Both arms return a `Session`.
- `db.add_source(name, *, url, format)` — `format ∈ {"parquet","csv","json"}`; url may be local or `s3://`/`gs://`/`azure://`.
- `db.with_tenant(tenant_id)` · `db.tenant() -> Optional[str]`
- `db.generate_embeddings(*, source, model, columns, key, modality=None) -> str` (returns embedding table name; `modality ∈ {"text","image","audio"}`).
- `db.sql(query) -> pyarrow.Table` · `db.encode_query(*, model, query, modality=None) -> list[float]`

**Tier 01 — Construct**
- `db.build_neighbor_graph(source, *, k, min_similarity=None, mutual=False, exact=False, table=None) -> str` → edge table `(src, dst, rank, similarity)`. **Use `exact=True` in the cookbook** (determinism).

**Tier 02 — Analyze**
- `db.search(source, *, query, k, filter=None, select=None) -> pyarrow.Table`
- `db.assemble_context(source, *, query, k, value_columns=None, aggregator=None, exclude_self=True, exclude_key=None, split=None, edge_source=None, edge_src_column=None, edge_dst_column=None, edge_type_column=None, edge_weight_column=None, edge_hops=None, edge_fanout=None, edge_direction=None, edge_types=None, min_weight=None, hybrid=False) -> dict` (keys: `context_vector`, `context_size`, `context_keys`, `value_rows`, `source`). `aggregator ∈ {"mean","sum","max"}`; `edge_direction ∈ {"out","in","undirected"}`.
- `db.rrf_fuse(ranked_lists, *, k_rrf=None) -> list[(str, float)]`
- `db.propagate_embeddings(source, *, embedding_table=None, edge_graph_table=None, edge_source=None, edge_src_column=None, edge_dst_column=None, edge_weight_column=None, direction=None, hops=None, weighting=None, alpha=None, output=None) -> str` — propagate an embedding table's features over a declared graph; returns a new searchable embedding table (`derived_from` the source). Pass **either** `edge_graph_table` (an S9 `neighbor_graph`) **or** `edge_source` (a registered edge table; cols default `"src"`/`"dst"`). `direction ∈ {"out"(default),"in","undirected"}`; `hops` default 2, **clamped to [1,3]**; `weighting ∈ {"degree_normalized"(default, symmetric-normalized + α-teleport = APPNP), "uniform"(random-walk mean ≈ SGC), "edge_similarity"(weighted by `edge_weight_column`)}`; `alpha` = APPNP teleport (default 0.1); `output ∈ {"final"(default, d-dim), "jumping_knowledge"(concat all hops, (K+1)·d)}`. **Deterministic** by construction (fixed `(group,neighbour)` f64 fold order — byte-identical across threads; no seeding). Refuses if the edge set exceeds ~2M.

**Tier 03 — Learn**
- `db.fine_tune(*, source, base_model, columns, method, embedding_loss=None, mnrl_temperature=None, mine_hard_negatives=None, lora_rank=None, epochs=None, batch_size=None, learning_rate=None, matryoshka_dims=None, ...) -> FineTuneJob` (`method ∈ {"triplet","pair"}`; `embedding_loss ∈ {"cosent","angle","cosine_mse","triplet","mnrl"}`).
- `db.fine_tune_graph(*, node_source, id_column, text_column, edge_source, src_column, dst_column, base_model, edge_provenance="declared", walk_length=None, walks_per_node=None, return_p=None, in_out_q=None, graph_hard_negatives=None, exclude_hops=None, embedding_loss=None, epochs=None, ...) -> FineTuneJob` (`edge_provenance ∈ {"declared","similarity"}` — **use `"declared"`** per the circularity contract).
- `job.wait()` · `job.status()` · `job.model_id`
- Eval: `db.eval_embeddings(*, source, golden_source, model=None, k=10, cohorts=None) -> dict` · `db.eval_compare(*, embedding_tables, source, golden_source, k=10) -> dict` · `db.eval_inference(...)` · `db.eval_per_query(eval_run_id) -> list`

**Tier 04 — Predict & Quantify**
- `db.train_context_predictor(source, *, key_column, task_column, value_column, architecture="attncnp", output="gaussian", objective="crps", context_k=32, hidden_dim=64, num_heads=4, num_layers=2, levels=None, beta=0.5, epochs=100, learning_rate=0.005, grad_clip=1.0, test_task_fraction=0.2, min_task_count=4, seed=0, model_id=None) -> str` (`architecture ∈ {"cnp","attncnp","tnp"}`; `output ∈ {"gaussian","quantile"}`; `objective ∈ {"crps","nll","betanll"}`).
- `db.predict_with_context_predictor(model_id, *, source, target_key, split=None, edge_source=None, edge_src_column=None, edge_dst_column=None, edge_type_column=None, edge_weight_column=None, edge_hops=None, edge_fanout=None, edge_direction=None, edge_types=None, min_weight=None, hybrid_ann_k=None) -> dict` (gaussian: `{"kind","mean","std"}`; quantile: `{"kind","levels"}`; plus `"source"`, `"context_ref"`). **The declared-edge params make this graph-conditioned — this is the tier-04 BYOG surface.**
- Conformal (**marginal only** in Python, by design): `db.conformalize(calibration, true_labels, test, *, alpha, score=None, raps_params=None) -> list[list[int]]` (sets; `score ∈ {"lac","aps","raps"}`); `db.conformalize_interval(predictions, observed, test_predictions, *, alpha) -> list[(float,float)]`; `db.conformalize_cqr(lower, upper, observed, test_lower, test_upper, *, alpha) -> list[(float,float)]`.
- Calibration eval (R2): `db.eval_calibration(*, source, golden_source, shape, cohorts=None) -> dict` (`shape ∈ {"gaussian","sample"}`; returns CRPS/NLL/adaptive_ece/sharpness/coverage).

**Not available in Python (do not call):** `search_by_id` (use `search` with a fetched vector); standalone Mondrian/weighted conformal (Rust/governance-only — the tier-04 repair is computed inline; see K0 §4). Provenance channels: `db.register_channel` / `db.add_channel_columns` exist if a recipe needs a custom evidence channel.

## 6. Success criteria

`quarto render` builds an empty-but-valid book; `jammi_cookbook` imports clean; `ci.yml` runs `--execute` and the no-deferral grep on an empty chapter set and passes; `CLAUDE.md` present; the API reference above is embedded in the repo (e.g. `chapters/_api.qmd` or `jammi_cookbook/_api_reference.md`) for chapter authors. Lint/format clean.
