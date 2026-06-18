# `jammi_ai` API reference (grounded, pinned `jammi_ai==0.31.0`)

These are the `jammi_ai` Python signatures the chapters call, **confirmed by
introspecting the installed `0.31.0` wheel** — `inspect.signature` resolved on a
live `connect("file://…")` instance, the way a caller invokes a verb through the
thin `Database` wrapper's composition (explicit methods on the wrapper, every
other verb forwarded to the native handle via `__getattr__`) — not transcribed
from a spec. Call them exactly as written. The cookbook pins this one version; if
the pin moves, re-introspect and update this file
(`scripts/check_api_reference.py` fails CI if a signature drifts).

## Two planes, one verb surface

`0.31.0` splits provisioning from execution, and the book's teaching spine is the
**`connect(target)` parity**: write a recipe once, swap only the target.

- **Control plane** — provisioning (register sources/topics/mutable tables,
  `list_*`, `describe_*`). In a deployment this is the `jammi` CLI; in the book it
  is the same `Database`/`RemoteDatabase` handle.
- **Data plane** — the ML/data verbs (`generate_embeddings`, `propagate_embeddings`,
  `search`, `fine_tune*`, `train_context_predictor`, `predict*`, `eval_*`,
  `conformalize*`).

`connect(target)` selects the transport **once** and returns a handle whose verb
surface is identical across planes:

- `jammi_ai.connect("file:///path")  -> Database` — embedded, in-process engine
  (CPU). The committed cache and CI read the book on this arm.
- `jammi_ai.connect("grpc://host:port") -> RemoteDatabase` — the pure-Python
  client (`jammi_client`) over a running `jammi-server`. The **GPU** compute tier
  (embedding, fine-tune, context-predictor training) runs here; the CPU embed
  wheel cannot do GPU.

A recipe written against either handle runs unchanged against the other — the
keystone's heavy emit connects `grpc://` to the GPU server; every chapter loads
the committed cache and runs `file://` on CPU.

## Setup / sources

- `jammi_ai.connect(target) -> Union[Database, RemoteDatabase]` — open a session against a target; `target` is a URL string (`file://` local, `grpc://`/`grpcs://` remote) or a parsed `Target`. Selects transport once.
- `jammi_ai.open_local(*, config=None, artifact_dir=None, gpu_device=None, inference_batch_size=None) -> Database` — the embedded engine directly (equivalent to `connect("file://<artifact_dir>")`).
- `jammi_ai.parse_target(target) -> Target` · `LocalTarget(artifact_dir)` / `RemoteTarget(endpoint, tls)`.
- `db.add_source(name, *, url, format)` — register a file-shaped source; `format ∈ {"parquet","csv","json"}`; `url` may be local or `s3://`/`gs://`/`azure://`. Re-registering a name raises (`Source already registered`).
- `db.list_sources() -> list[dict]` · `db.describe_source(source_id) -> Optional[dict]` — control-plane catalog reads.
- `db.get_server_info() -> dict` — `{"version", "features", "storage_backends", "services"}`.
- `db.set_tenant(tenant_id) -> None` · `db.tenant_scope(tenant_id)` (context manager) · `db.tenant() -> Optional[str]` — `set_tenant` binds the tenant scope to the connection *in place* (pass `""` to clear); `tenant_scope` binds it for a `with` block and restores the prior scope on exit; `tenant` reads the current scope. The bound scope filters catalog listing (`list_sources`) and row reads from sources carrying a `tenant_id` discriminator column to `tenant_id = $cur OR IS NULL`; it does **not** gate a discriminator-less source (the engine does not authenticate). `tenant_id` is an opaque UUID.
- `db.generate_embeddings(*, source, model, columns, key, modality=None, cache=None) -> str` — returns the embedding-table name; `modality ∈ {"text","image","audio"}`. `model` is a model-id string resolved from HuggingFace Hub. `cache` (H4) opts the call into the engine's incremental-recompute cache: a recorded producing descriptor lets a later `recompute`/`staleness`/`verify_materialization` reason about this output. (Runs on the GPU server when connected `grpc://`.)
- `db.sql(query) -> pyarrow.Table` · `db.encode_query(*, model, query, modality=None) -> list[float]`
- `db.rrf_fuse(ranked_lists, *, k_rrf=None) -> list[(str, float)]`

A registered file source is queried as `<source>.public.<source>`
(catalog.schema.table). An engine-produced embedding table is queried as
`"jammi.<table>"` (a quoted single identifier).

## Tier 01 — Construct

- `db.build_neighbor_graph(source, *, k, min_similarity=None, mutual=False, exact=False, table=None, cache=None) -> str` — **`source` is the registered source name** (e.g. `"arxiv_papers"`), not an embedding-table name; the engine discovers that source's ready embedding table and builds the self-kNN graph. Returns an edge table `(src, dst, rank, similarity)`. **Use `exact=True`** in the cookbook (determinism). `cache` (H4) records a producing descriptor for incremental recompute.

## Tier 02 — Analyze

- `db.search(source, *, query, k, filter=None, select=None) -> pyarrow.Table` — `query` is a vector (`list[float]`).
- `db.assemble_context(source, *, query, k, value_columns=None, aggregator=None, exclude_self=True, exclude_key=None, split=None, edge_source=None, edge_src_column=None, edge_dst_column=None, edge_type_column=None, edge_weight_column=None, edge_hops=None, edge_fanout=None, edge_direction=None, edge_types=None, min_weight=None, hybrid=False) -> dict` — keys `context_vector`, `context_size`, `context_keys`, `value_rows`, `source`. `aggregator ∈ {"mean","sum","max"}`; `edge_direction ∈ {"out","in","undirected"}`.
- `db.propagate_embeddings(source, *, embedding_table=None, edge_graph_table=None, edge_source=None, edge_src_column=None, edge_dst_column=None, edge_weight_column=None, direction=None, hops=None, weighting=None, alpha=None, output=None, cache=None) -> str` — propagate an **embedding table's** features over a declared graph; returns a new searchable embedding table. Pass `embedding_table` (the name `generate_embeddings` returned) plus **either** `edge_graph_table` (an S9 `neighbor_graph`) **or** `edge_source` (a registered edge table; cols default `"src"`/`"dst"`). `direction ∈ {"out"(default),"in","undirected"}`; `hops` default 2, **clamped to [1,3]**; `weighting ∈ {"degree_normalized"(default; symmetric-normalized + α-teleport = APPNP), "uniform"(random-walk mean ≈ SGC), "edge_similarity"(weighted by edge_weight_column)}`; `alpha` = APPNP teleport (default 0.1); `output ∈ {"final"(default, d-dim), "jumping_knowledge"(concat all hops, (K+1)·d)}`. `cache` (H4) records a producing descriptor for incremental recompute. **Deterministic** by construction (fixed `(group,neighbour)` f64 fold order — byte-identical across threads; no seeding). Refuses if the edge set exceeds ~2M.

## Tier 03 — Learn

- `db.fine_tune(*, source, base_model, columns, method, task=None, lora_rank=None, lora_alpha=None, lora_dropout=None, learning_rate=None, epochs=None, batch_size=None, max_seq_length=None, validation_fraction=None, early_stopping_patience=None, warmup_steps=None, gradient_accumulation_steps=None, triplet_margin=None, target_modules=None, early_stopping_metric=None, backbone_dtype=None, weight_decay=None, max_grad_norm=None, embedding_loss=None, mnrl_temperature=None, cached=None, mine_hard_negatives=None, hard_negative_k=None, hard_negative_exclude_hops=None, hard_negative_refresh_every=None, matryoshka_dims=None, seed=None, regression_loss=None, regression_beta=None, quantile_levels=None) -> TrainingJob` — **`method` is required** (the adapter family; only `"lora"` is supported). `task` is the optional `ModelTask` string (`"text_embedding"`, `"classification"`, `"regression"`, …). `embedding_loss ∈ {"cosent","angle","cosine_mse","triplet","mnrl"}`. `seed` pins fine-tune determinism. **Regression objective** (`task="regression"` only): `regression_loss ∈ {"gaussian_nll","beta_nll","crps","pinball"}` — default β-NLL (a stop-gradient σ^{2β} re-weighting that prevents mean-starvation in joint μ,σ² NLL; distinct from the σ-floor overconfidence guard); `regression_beta` is β for `beta_nll` (default 0.5); `quantile_levels` is required for `pinball`. The supervision source must project columns named exactly `text` and `target` (numeric); served predictions are de-standardized to raw outcome units (`predicted_mean`/`predicted_std`).
- `db.fine_tune_graph(*, node_source, id_column, text_column, edge_source, src_column, dst_column, base_model, edge_provenance="declared", walk_length=None, walks_per_node=None, return_p=None, in_out_q=None, graph_hard_negatives=None, exclude_hops=None, min_negatives=None, sample_seed=None, embedding_loss=None, mnrl_temperature=None, epochs=None, batch_size=None, learning_rate=None, lora_rank=None, matryoshka_dims=None) -> TrainingJob` — `edge_provenance ∈ {"declared","similarity"}`; **use `"declared"`** per the circularity contract.
- `TrainingJob`: `job.wait() -> None` · `job.status() -> str` · `job.model_id` · `job.job_id`. (Remote handles return `RemoteTrainingJob` with the same surface.)
- Eval (R1): `db.eval_embeddings(*, source, golden_source, model=None, k=10, cohorts=None) -> dict` · `db.eval_compare(*, embedding_tables, source, golden_source, k=10) -> dict` · `db.eval_inference(*, model, source, columns, task, golden_source, label_column) -> dict` · `db.eval_per_query(eval_run_id) -> list`

## Tier 04 — Predict & Quantify

- `db.train_context_predictor(source, *, key_column, task_column, value_column, architecture="attncnp", output="gaussian", objective="crps", context_k=32, hidden_dim=64, num_heads=4, num_layers=2, levels=None, beta=0.5, epochs=100, learning_rate=0.005, grad_clip=1.0, test_task_fraction=0.2, min_task_count=4, seed=0, model_id=None) -> TrainingJob` — `architecture ∈ {"cnp","attncnp","tnp"}`; `output ∈ {"gaussian","quantile"}`; `objective ∈ {"crps","nll","betanll"}`. Returns a `TrainingJob`; `job.model_id` after `job.wait()` is the predictor id.
- `db.predict_with_context_predictor(model_id, *, source, target_key, split=None, edge_source=None, edge_src_column=None, edge_dst_column=None, edge_type_column=None, edge_weight_column=None, edge_hops=None, edge_fanout=None, edge_direction=None, edge_types=None, min_weight=None, hybrid_ann_k=None) -> dict` — gaussian: `{"kind","mean","std"}`; quantile: `{"kind","levels"}`; plus `"source"` and `"context_ref"`. **The declared-edge params make this graph-conditioned — the tier-04 BYOG surface.**
- Conformal (**marginal only** in Python, by design): `db.conformalize(calibration, true_labels, test, *, alpha, score=None, raps_params=None) -> list[list[int]]` (`score ∈ {"lac","aps","raps"}`); `db.conformalize_interval(predictions, observed, test_predictions, *, alpha) -> list[(float,float)]`; `db.conformalize_cqr(lower, upper, observed, test_lower, test_upper, *, alpha) -> list[(float,float)]`.
- Calibration eval (R2): `db.eval_calibration(*, source, golden_source, shape, cohorts=None) -> dict` — `shape ∈ {"gaussian","sample"}`; returns `aggregate` (CRPS/NLL/adaptive_ece/sharpness/coverage), `per_cohort`, `per_record`. `golden_source` resolves in the engine's default catalog, so pass the **fully-qualified path** `"<source>.public.<source>"`, not the bare source id (the bare id raises `table not found`). For `"gaussian"` the golden table carries `record_id, mean, sd, outcome`.

## Model catalog (control plane)

A `fine_tune*` / `train_context_predictor` job registers a model under its
`job.model_id`; these control-plane verbs let you **see** the models the engine
resolves and trains, and **clean them up**. Pre-trained models are served by **id**
(the resolver loads an HF or local model by reference). Scoped to the session's
bound tenant. Each dict carries `{model_id, backend, task, status}`.

- `db.list_models() -> list[dict]` — one record per model in the catalog for the current tenant (peer of `list_sources`).
- `db.describe_model(model_id) -> Optional[dict]` — describe one model by id, or `None` if it is not in the catalog.
- `db.delete_model(model_id, *, version=None, if_exists=False) -> None` — remove the model row from the catalog. Refused with `FAILED_PRECONDITION` while any reference still points at the model. `version=None` targets the latest version; `if_exists=True` makes a missing model a no-op (the flag rides the request — the server is authoritative).

## Provenance channels

- `db.register_channel(channel_id, *, priority, columns)` · `db.add_channel_columns(channel_id, *, columns)` — for a recipe that needs a custom evidence channel.

## Mutable companion tables

- `db.create_mutable_table(name, *, schema, primary_key, indexes=None, order_column=None, chunk_size=None) -> str` — provision a mutable companion table alongside the append-only result tables; `schema` is the column definition, `primary_key` the upsert key. `indexes` declares secondary lookups, `order_column` the sort key, `chunk_size` the row-group size.
- `db.drop_mutable_table(name, *, if_exists=False) -> None` — drop a mutable table; `if_exists=True` makes a missing table a no-op.
- `db.list_mutable_tables() -> list[dict]` — control-plane catalog read of the registered mutable tables.

## Trigger stream / topics

- `db.register_topic(name, *, schema, broker_metadata=None) -> str` — register a trigger-stream topic with a row `schema`; `broker_metadata` carries backend-specific configuration.
- `db.drop_topic(name, *, if_exists=False) -> None` — drop a topic; `if_exists=True` makes a missing topic a no-op.
- `db.list_topics() -> list[str]` — control-plane catalog read of the registered topic names.
- `db.publish_topic(topic, *, batch) -> int` — publish an Arrow `batch` of rows onto a topic; returns the 0-based offset the batch landed at.
- `db.subscribe_collect(topic, *, predicate=None, from_offset=None, max_batches=64) -> pyarrow.Table` — replay a topic from `from_offset` into one table; `predicate` is a SQL filter over the topic schema (a batch whose rows all filter out is dropped, not yielded). `max_batches` is the **terminator**: the call replays the backing table then tails the live broker, so it returns synchronously only when `max_batches` equals the number of yielded batches available from `from_offset` (`num_published - from_offset` unfiltered; the matching-batch count under a predicate). A larger `max_batches` blocks on the broker tail.

## Point-in-time join + materialization / incremental recompute (H4, `0.31.0`)

The H4 horizon adds a point-in-time temporal join and a read-only
materialization / lineage / incremental-recompute surface. The cache-aware verbs
above (`generate_embeddings`, `infer`, `build_neighbor_graph`,
`propagate_embeddings`) accept a `cache` kwarg that records a **producing
descriptor** for their output table; these verbs then reason over that descriptor.

- `db.asof_join(spine, facts, *, spine_by, spine_time, facts_by, facts_time, direction=None, boundary=None, tolerance_duration_micros=None, tolerance_steps=None, tie_break_column=None, project=None) -> str` — assemble a **point-in-time-correct** table: for each `spine` row, attach the `facts` row valid as-of the spine row's temporal key, within each equality group; returns the new table name (read it via `sql`). `spine`/`facts` are registered source ids; the `*_by`/`*_time` pairs name each side's equality + temporal columns (an empty `*_by` is one global group). `direction ∈ {"backward"(default, leakage-safe),"forward","nearest"}`; `boundary ∈ {"inclusive"(default),"exclusive"}`; **at most one** tolerance unit (`tolerance_duration_micros` *or* `tolerance_steps`) may be given; `tie_break_column` names the secondary descending disambiguator (when omitted, a duplicate at the matched instant fails loudly). Left rows are always preserved; unmatched fact columns are null. Mirrors `RemoteDatabase.asof_join`.
- `db.verify_materialization(table, expected_definition=None) -> dict` — recompute a materialised result table's artifact digest (and, if given, an expected definition hash) against its `.materialization.json` manifest. Returns `{"verdict": "match" | "mismatch" | "match_with_unpinned_inputs" | "missing_manifest", ...}`. **Read-only** — attests the Parquet data (not the ANN index); never acts on a verdict.
- `db.staleness(table, current_definition) -> dict` — the read-only staleness sensor: is a `ready` table still the output of its recorded definition over its inputs' current state? `current_definition` is the hash of how the table is produced now. Returns `{"staleness": "fresh" | "stale" | "undecidable" | "missing_manifest", ...}` (`stale` carries `reasons`; `undecidable` carries `unpinned` + `decided_reasons`). **Read-only.**
- `db.derives_from(table) -> list[dict]` — the one-hop reverse-dependency edges of a result table: every `ready` table that anchored on it, as `{"input", "derived", "kind"}`. Read-only lineage the caller walks transitively.
- `db.recompute(table, *, cascade=None) -> dict` — re-invoke a result table's recorded producer over its inputs' current state. `cascade ∈ {"report_only"(default — recompute the named table only and report the transitive downstream-stale set), "downstream"(additionally sweep every transitive dependent once, in dependency order)}`. A pre-contract table (no recorded producing descriptor) raises. Returns `{"recomputed": [{"original","recomputed","outcome"}], "downstream_stale": [...]}`; read each recomputed table via `sql`. Mirrors `RemoteDatabase.recompute`.

## Not available in Python (do not call)

- `search_by_id` — use `search` with a fetched vector.
- Standalone Mondrian / weighted conformal — not in the OSS Python surface. The
  tier-04 graph-aware repair is computed **inline in the notebook** (~10 lines of
  numpy); a productionized, governed cohort surface is out of scope. OSS Python
  exposes only the **marginal** `conformalize*` surface (K0 §4).
