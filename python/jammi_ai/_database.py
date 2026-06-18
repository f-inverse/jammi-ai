"""The embedded `Database`: a thin Python wrapper over the `_NativeDatabase` handle.

`connect("file://…")` returns one of these. It is the user-facing embedded
surface, composed over the compiled `_native._NativeDatabase` low-level handle:

* Every verb whose request is still assembled in Rust is forwarded verbatim to
  the native handle by ``__getattr__`` — the embedded implementation is unchanged.
* The migrated verbs — the training verbs (`fine_tune`, `fine_tune_graph`,
  `train_context_predictor`), the bulk inference verb (`infer`), the
  engine-state pipeline verbs (`build_neighbor_graph`, `propagate_embeddings`,
  `assemble_context`), the embedding + search verbs (`generate_embeddings`,
  `encode_query`, `search`), the catalog/substrate verbs (`register_channel`,
  `add_channel_columns`, `create_mutable_table`, `register_topic`), and the eval
  verbs (`eval_embeddings`, `eval_per_query`, `eval_inference`, `eval_compare`,
  `eval_calibration`) — are explicit
  methods here. They build their request
  with the SAME pure-Python assembly the remote client uses
  (`jammi_client._assembly.build_*_request`), serialize it, and hand the bytes to
  the native handle's `_*_proto` primitive — which decodes through the engine's
  shared wire seam and runs the verb in-process. So the embedded and remote
  paths share one request assembly and one decode, differing only in the
  transport primitive (a PyO3 call here, a gRPC call there). The signatures
  mirror `jammi_client.RemoteDatabase`'s — the conformance contract.

The kwargs→struct assembly the native handle used to carry for these verbs is
gone from Rust; it lives once in `jammi_client._assembly`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pyarrow as pa

from jammi_client._assembly import (
    build_add_channel_columns_request,
    build_asof_join_request,
    build_assemble_context_request,
    build_context_predictor_request,
    build_create_mutable_table_request,
    build_encode_query_request,
    build_eval_calibration_request,
    build_eval_compare_request,
    build_eval_embeddings_request,
    build_eval_inference_request,
    build_eval_per_query_request,
    build_fine_tune_graph_request,
    build_fine_tune_request,
    build_generate_embeddings_request,
    build_infer_request,
    build_neighbor_graph_request,
    build_propagate_embeddings_request,
    build_recompute_request,
    build_register_channel_request,
    build_register_topic_request,
    build_search_request,
    recompute_report_to_dict,
)
from jammi_client._generated.jammi.v1 import pipeline_pb2


class Database:
    """The embedded engine, wrapping a compiled `_NativeDatabase` handle.

    Holds the low-level native handle by composition. Verbs not declared
    explicitly here forward to it through ``__getattr__``, so the full embedded
    vocabulary is reachable unchanged; the training verbs are explicit so they
    drive the shared request assembly and so the conformance guard can introspect
    them at the class level.
    """

    def __init__(self, native: object) -> None:
        # Stored under a private name; ``__getattr__`` forwards every other
        # attribute to it. Set via the instance dict directly so ``__getattr__``
        # never recurses while resolving `_native` itself.
        object.__setattr__(self, "_native", native)

    def __getattr__(self, name: str):
        # Reached only for attributes NOT found on the wrapper (so never for the
        # explicit training verbs, nor `_native` itself): forward to the native
        # handle, which carries every un-migrated verb's embedded implementation.
        return getattr(self._native, name)

    def __dir__(self) -> list:
        # The composed surface: the wrapper's own attributes (the explicit
        # training verbs) UNION the native handle's, so introspection and REPL
        # completion see every verb a caller can invoke — both the explicit ones
        # here and the ones `__getattr__` forwards. Without this, `dir()` lists
        # only the wrapper's statics and hides the delegated verbs.
        return sorted(set(super().__dir__()) | set(dir(self._native)))

    def fine_tune(
        self,
        *,
        source: str,
        base_model: str,
        columns: List[str],
        method: str,
        task: Optional[str] = None,
        lora_rank: Optional[int] = None,
        lora_alpha: Optional[float] = None,
        lora_dropout: Optional[float] = None,
        learning_rate: Optional[float] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        validation_fraction: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        warmup_steps: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        triplet_margin: Optional[float] = None,
        target_modules: Optional[List[str]] = None,
        early_stopping_metric: Optional[str] = None,
        backbone_dtype: Optional[str] = None,
        weight_decay: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        embedding_loss: Optional[str] = None,
        mnrl_temperature: Optional[float] = None,
        cached: Optional[bool] = None,
        mine_hard_negatives: Optional[bool] = None,
        hard_negative_k: Optional[int] = None,
        hard_negative_exclude_hops: Optional[int] = None,
        hard_negative_refresh_every: Optional[int] = None,
        matryoshka_dims: Optional[List[int]] = None,
        seed: Optional[int] = None,
        regression_loss: Optional[str] = None,
        regression_beta: Optional[float] = None,
        quantile_levels: Optional[List[float]] = None,
    ):
        """Submit a LoRA fine-tuning job to the in-process engine; poll the handle.

        Returns a `TrainingJob` — same handle shape and verb signature as the
        remote `RemoteDatabase.fine_tune`. The request is assembled with the
        shared `FineTuneSpec` builder and submitted through the engine's wire
        seam; all config kwargs are optional, applying the engine defaults when
        omitted.
        """
        request = build_fine_tune_request(
            source=source,
            base_model=base_model,
            columns=columns,
            method=method,
            task=task,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            validation_fraction=validation_fraction,
            early_stopping_patience=early_stopping_patience,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            triplet_margin=triplet_margin,
            target_modules=target_modules,
            early_stopping_metric=early_stopping_metric,
            backbone_dtype=backbone_dtype,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            embedding_loss=embedding_loss,
            mnrl_temperature=mnrl_temperature,
            cached=cached,
            mine_hard_negatives=mine_hard_negatives,
            hard_negative_k=hard_negative_k,
            hard_negative_exclude_hops=hard_negative_exclude_hops,
            hard_negative_refresh_every=hard_negative_refresh_every,
            matryoshka_dims=matryoshka_dims,
            seed=seed,
            regression_loss=regression_loss,
            regression_beta=regression_beta,
            quantile_levels=quantile_levels,
        )
        return self._native._start_training_proto(request.SerializeToString())

    def fine_tune_graph(
        self,
        *,
        node_source: str,
        id_column: str,
        text_column: str,
        edge_source: str,
        src_column: str,
        dst_column: str,
        base_model: str,
        edge_provenance: str = "declared",
        walk_length: Optional[int] = None,
        walks_per_node: Optional[int] = None,
        return_p: Optional[float] = None,
        in_out_q: Optional[float] = None,
        graph_hard_negatives: Optional[int] = None,
        exclude_hops: Optional[int] = None,
        min_negatives: Optional[int] = None,
        sample_seed: Optional[int] = None,
        embedding_loss: Optional[str] = None,
        mnrl_temperature: Optional[float] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        lora_rank: Optional[int] = None,
        matryoshka_dims: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """Submit a graph-supervised fine-tune (S11) to the in-process engine.

        Returns a `TrainingJob`, mirroring the remote
        `RemoteDatabase.fine_tune_graph`. The request is assembled with the
        shared `GraphFineTuneSpec` builder (which carries the graph-only
        embedding-loss guard) and submitted through the engine's wire seam.
        `edge_provenance` is the load-bearing circularity distinction — "declared"
        external edges teach the metric something new; "similarity" edges are a
        weak bootstrap only.
        """
        request = build_fine_tune_graph_request(
            node_source=node_source,
            id_column=id_column,
            text_column=text_column,
            edge_source=edge_source,
            src_column=src_column,
            dst_column=dst_column,
            base_model=base_model,
            edge_provenance=edge_provenance,
            walk_length=walk_length,
            walks_per_node=walks_per_node,
            return_p=return_p,
            in_out_q=in_out_q,
            graph_hard_negatives=graph_hard_negatives,
            exclude_hops=exclude_hops,
            min_negatives=min_negatives,
            sample_seed=sample_seed,
            embedding_loss=embedding_loss,
            mnrl_temperature=mnrl_temperature,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
            matryoshka_dims=matryoshka_dims,
            seed=seed,
        )
        return self._native._start_training_proto(request.SerializeToString())

    def train_context_predictor(
        self,
        source: str,
        *,
        key_column: str,
        task_column: str,
        value_column: str,
        architecture: str = "attncnp",
        output: str = "gaussian",
        objective: str = "crps",
        context_k: int = 32,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        levels: Optional[List[float]] = None,
        beta: float = 0.5,
        epochs: int = 100,
        learning_rate: float = 0.005,
        grad_clip: float = 1.0,
        test_task_fraction: float = 0.2,
        min_task_count: int = 4,
        seed: int = 0,
        model_id: Optional[str] = None,
    ):
        """Submit an amortized in-context predictor (S19) meta-training to the
        in-process engine.

        Returns a `TrainingJob`, mirroring the remote
        `RemoteDatabase.train_context_predictor`. The request is assembled with
        the shared `ContextPredictorSpec` builder (which builds the
        gaussian/quantile predictive head and applies the `output='quantile'
        requires levels` check) and submitted through the engine's wire seam.
        """
        request = build_context_predictor_request(
            source,
            key_column=key_column,
            task_column=task_column,
            value_column=value_column,
            architecture=architecture,
            output=output,
            objective=objective,
            context_k=context_k,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            levels=levels,
            beta=beta,
            epochs=epochs,
            learning_rate=learning_rate,
            grad_clip=grad_clip,
            test_task_fraction=test_task_fraction,
            min_task_count=min_task_count,
            seed=seed,
            model_id=model_id,
        )
        return self._native._start_training_proto(request.SerializeToString())

    def infer(
        self,
        *,
        source: str,
        model: str,
        columns: List[str],
        task: str,
        key: str,
        cache: Optional[str] = None,
    ) -> pa.Table:
        """Run `model` over `columns` of a registered source for `task`,
        returning one output row per source row as a `pyarrow.Table`.

        `task` is the snake-case model-task string (``"text_embedding"``,
        ``"image_embedding"``, ``"audio_embedding"``, ``"classification"``,
        ``"ner"``, ``"regression"``); `key` names the column whose value becomes
        each output row's ``_row_id``. `cache` opts into memoization
        (``"use"``) or keeps the default recompute (``None``/``"bypass"``) —
        inference anchors its source unpinned, so ``"use"`` is honestly always a
        recompute today. Same handle shape and verb signature as the remote
        `RemoteDatabase.infer`. The request is assembled with the shared
        `InferRequest` builder and submitted through the engine's wire seam.
        """
        request = build_infer_request(
            source=source,
            model=model,
            columns=columns,
            task=task,
            key=key,
            cache=cache,
        )
        return self._native._infer_proto(request.SerializeToString())

    def build_neighbor_graph(
        self,
        source: str,
        *,
        k: int,
        min_similarity: Optional[float] = None,
        mutual: bool = False,
        exact: bool = False,
        table: Optional[str] = None,
        cache: Optional[str] = None,
    ) -> str:
        """Materialise the k-NN graph of a source's embedding table and return the
        new edge table's name.

        The returned table has columns ``(src, dst, rank, similarity)``. The
        default driver is index-assisted and approximate; pass ``exact=True`` for
        a deterministic, complete graph. ``min_similarity`` floors weak edges;
        ``mutual=True`` keeps only reciprocal edges. Mirrors the remote
        `RemoteDatabase.build_neighbor_graph`; the request is assembled with the
        shared `BuildNeighborGraphRequest` builder and submitted through the
        engine's wire seam. Read the table via :meth:`sql`.
        """
        request = build_neighbor_graph_request(
            source,
            k=k,
            min_similarity=min_similarity,
            mutual=mutual,
            exact=exact,
            table=table,
            cache=cache,
        )
        return self._native._build_neighbor_graph_proto(request.SerializeToString())

    def propagate_embeddings(
        self,
        source: str,
        *,
        embedding_table: Optional[str] = None,
        edge_graph_table: Optional[str] = None,
        edge_source: Optional[str] = None,
        edge_src_column: Optional[str] = None,
        edge_dst_column: Optional[str] = None,
        edge_weight_column: Optional[str] = None,
        direction: Optional[str] = None,
        hops: Optional[int] = None,
        weighting: Optional[str] = None,
        alpha: Optional[float] = None,
        output: Optional[str] = None,
        cache: Optional[str] = None,
    ) -> str:
        """Propagate an embedding table's features over a declared graph (the
        decoupled-GNN forward pass) into a new, searchable embedding table.

        The graph is either an S9 similarity graph (``edge_graph_table``, a
        :meth:`build_neighbor_graph` output) or a registered external edge source
        (``edge_source``) — pass exactly one. ``weighting`` selects the neighbour
        normalisation; ``output`` is ``"final"`` or ``"jumping_knowledge"``.
        Returns the materialised table's name. Mirrors the remote
        `RemoteDatabase.propagate_embeddings`; the request is assembled with the
        shared `PropagateEmbeddingsRequest` builder and submitted through the
        engine's wire seam. Read the table via :meth:`sql`.
        """
        request = build_propagate_embeddings_request(
            source,
            embedding_table=embedding_table,
            edge_graph_table=edge_graph_table,
            edge_source=edge_source,
            edge_src_column=edge_src_column,
            edge_dst_column=edge_dst_column,
            edge_weight_column=edge_weight_column,
            direction=direction,
            hops=hops,
            weighting=weighting,
            alpha=alpha,
            output=output,
            cache=cache,
        )
        return self._native._propagate_embeddings_proto(request.SerializeToString())

    def asof_join(
        self,
        spine: str,
        facts: str,
        *,
        spine_by: List[str],
        spine_time: str,
        facts_by: List[str],
        facts_time: str,
        direction: Optional[str] = None,
        boundary: Optional[str] = None,
        tolerance_duration_micros: Optional[int] = None,
        tolerance_steps: Optional[int] = None,
        tie_break_column: Optional[str] = None,
        project: Optional[List[str]] = None,
    ) -> str:
        """Assemble a point-in-time-correct table: for each row of ``spine``,
        attach the ``facts`` row valid as-of the spine row's temporal key, within
        each equality group, and return the new table's name.

        ``spine``/``facts`` are registered source ids; the ``*_by``/``*_time``
        pairs name each side's equality + temporal columns (an empty ``*_by`` is
        one global group). ``direction`` is ``"backward"`` (default, leakage-
        safe), ``"forward"``, or ``"nearest"``; ``boundary`` is ``"inclusive"``
        (default) or ``"exclusive"``; at most one tolerance unit may be given;
        ``tie_break_column`` names the secondary descending disambiguator, and
        when omitted a duplicate at the matched instant fails loudly. Left rows
        are always preserved; unmatched fact columns are null. Mirrors the remote
        `RemoteDatabase.asof_join`; the request is assembled with the shared
        `AsofJoinRequest` builder and submitted through the engine's wire seam.
        Read the table via :meth:`sql`.
        """
        request = build_asof_join_request(
            spine,
            facts,
            spine_by=spine_by,
            spine_time=spine_time,
            facts_by=facts_by,
            facts_time=facts_time,
            direction=direction,
            boundary=boundary,
            tolerance_duration_micros=tolerance_duration_micros,
            tolerance_steps=tolerance_steps,
            tie_break_column=tie_break_column,
            project=project,
        )
        return self._native._asof_join_proto(request.SerializeToString())

    def recompute(
        self,
        table: str,
        *,
        cascade: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Re-invoke a result table's recorded producer over the inputs' current
        state and return the recompute report.

        ``table`` is the result table to recompute; ``cascade`` is
        ``"report_only"`` (default — recompute the named table only and report the
        transitive downstream-stale set) or ``"downstream"`` (additionally sweep
        every transitive dependent once, in dependency order). A pre-contract
        table (no recorded producing descriptor) raises. Mirrors the remote
        `RemoteDatabase.recompute`; the request is assembled with the shared
        `RecomputeRequest` builder and submitted through the engine's wire seam.
        The report is a dict
        ``{"recomputed": [{"original", "recomputed", "outcome"}], "downstream_stale": [...]}``.
        Read each recomputed table via :meth:`sql`.
        """
        request = build_recompute_request(table, cascade=cascade)
        report = pipeline_pb2.RecomputeReport()
        report.ParseFromString(self._native._recompute_proto(request.SerializeToString()))
        return recompute_report_to_dict(report)

    def assemble_context(
        self,
        source: str,
        *,
        query: List[float],
        k: int,
        value_columns: Optional[List[str]] = None,
        aggregator: Optional[str] = None,
        exclude_self: bool = True,
        exclude_key: Optional[str] = None,
        split: Optional[str] = None,
        edge_source: Optional[str] = None,
        edge_src_column: Optional[str] = None,
        edge_dst_column: Optional[str] = None,
        edge_type_column: Optional[str] = None,
        edge_weight_column: Optional[str] = None,
        edge_hops: Optional[int] = None,
        edge_fanout: Optional[int] = None,
        edge_direction: Optional[str] = None,
        edge_types: Optional[List[str]] = None,
        min_weight: Optional[float] = None,
        hybrid: bool = False,
    ) -> Dict[str, Any]:
        """Assemble and encode a target's context set: retrieve `k` nearest
        neighbours of `query` (or a declared-edge walk / their union), pair them
        with `value_columns`, and pool the neighbour vectors into one fixed-width
        context vector.

        Returns the same dict shape as the remote `RemoteDatabase`:
        ``context_vector`` (list of floats, or ``None`` for a degenerate empty
        context), ``context_size``, ``context_keys``, ``value_rows`` (a
        ``pyarrow.Table``), and ``source`` (the assembly fact). The request is
        assembled with the shared `AssembleContextRequest` builder and submitted
        through the engine's wire seam.
        """
        request = build_assemble_context_request(
            source,
            query=query,
            k=k,
            value_columns=value_columns,
            aggregator=aggregator,
            exclude_self=exclude_self,
            exclude_key=exclude_key,
            split=split,
            edge_source=edge_source,
            edge_src_column=edge_src_column,
            edge_dst_column=edge_dst_column,
            edge_type_column=edge_type_column,
            edge_weight_column=edge_weight_column,
            edge_hops=edge_hops,
            edge_fanout=edge_fanout,
            edge_direction=edge_direction,
            edge_types=edge_types,
            min_weight=min_weight,
            hybrid=hybrid,
        )
        return self._native._assemble_context_proto(request.SerializeToString())

    def encode_query(
        self,
        *,
        model: str,
        query: Union[str, bytes],
        modality: Optional[str] = None,
    ) -> List[float]:
        """Encode a single query into an embedding vector with the given model.

        `query` is a string for the text tower or raw bytes for the image/audio
        tower; `modality` selects the tower (`"text"`/`"image"`/`"audio"`,
        defaulting to text). Same handle shape and verb signature as the remote
        `RemoteDatabase.encode_query`; the request is assembled with the shared
        `EncodeQueryRequest` builder and submitted through the engine's wire seam.
        """
        request = build_encode_query_request(
            model=model,
            query=query,
            modality=modality,
        )
        return self._native._encode_query_proto(request.SerializeToString())

    def generate_embeddings(
        self,
        *,
        source: str,
        model: str,
        columns: List[str],
        key: str,
        modality: Optional[str] = None,
        cache: Optional[str] = None,
    ) -> str:
        """Embed `columns` of a registered source, persisting one vector per row.

        `modality` selects the tower (`"text"`/`"image"`/`"audio"`, defaulting to
        text); `key` names the column whose value becomes each embedding row's
        key; `cache` opts into memoization (``"use"``) or keeps the default
        recompute (``None``/``"bypass"``) — embeddings anchor their source
        unpinned, so ``"use"`` is honestly always a recompute today. Returns the
        result table name. Same handle shape and verb signature as the remote
        `RemoteDatabase.generate_embeddings`; the request is assembled with the
        shared `GenerateEmbeddingsRequest` builder and submitted through the
        engine's wire seam.
        """
        request = build_generate_embeddings_request(
            source=source,
            model=model,
            columns=columns,
            key=key,
            modality=modality,
            cache=cache,
        )
        return self._native._generate_embeddings_proto(request.SerializeToString())

    def search(
        self,
        source: str,
        *,
        query: List[float],
        k: int,
        filter: Optional[str] = None,
        select: Optional[List[str]] = None,
        embedding_table: Optional[str] = None,
    ) -> pa.Table:
        """Nearest-neighbor search over a source's embedding table.

        `query` is the query vector; `filter` is an optional SQL predicate over
        the hydrated results; `select` projects columns (empty keeps the
        keyed+scored shape). `embedding_table` names which of the source's
        embedding tables to search (e.g. a raw, propagated, or fine-tuned table);
        ``None`` searches the most-recent ready table. Returns a `pyarrow.Table`.
        Mirrors the remote `RemoteDatabase.search`; the request is assembled with
        the shared `SearchRequest` builder and submitted through the engine's
        wire seam (only the request is shared — the Arrow response wrapping is the
        embedded transport's).
        """
        request = build_search_request(
            source,
            query=query,
            k=k,
            filter=filter,
            select=select,
            embedding_table=embedding_table,
        )
        return self._native._search_proto(request.SerializeToString())

    def register_channel(
        self,
        channel_id: str,
        *,
        priority: int,
        columns: Sequence[Tuple[str, str]],
    ) -> None:
        """Register an evidence-provenance channel and its initial columns.

        `columns` is a list of `(name, dtype)` tuples where `dtype` is the
        canonical PascalCase token (`"Float32"`, `"Utf8"`, …); `priority` orders
        the channel against others contributing the same column. The channel id is
        unique per tenant, scoped to the session's bound tenant. Same handle shape
        and verb signature as the remote `RemoteDatabase.register_channel`; the
        request is assembled with the shared `RegisterChannelRequest` builder and
        submitted through the engine's wire seam.
        """
        request = build_register_channel_request(
            channel_id,
            priority=priority,
            columns=columns,
        )
        self._native._register_channel_proto(request.SerializeToString())

    def add_channel_columns(
        self,
        channel_id: str,
        *,
        columns: Sequence[Tuple[str, str]],
    ) -> None:
        """Append columns to an already-registered channel (append-only).

        `columns` is a list of `(name, dtype)` tuples in the same PascalCase
        vocabulary as :meth:`register_channel`. The append-only invariant is
        enforced: redeclaring an existing column with a different dtype raises.
        Same handle shape and verb signature as the remote
        `RemoteDatabase.add_channel_columns`; the request is assembled with the
        shared `AddChannelColumnsRequest` builder and submitted through the
        engine's wire seam.
        """
        request = build_add_channel_columns_request(
            channel_id,
            columns=columns,
        )
        self._native._add_channel_columns_proto(request.SerializeToString())

    def create_mutable_table(
        self,
        name: str,
        *,
        schema: pa.Schema,
        primary_key: List[str],
        indexes: Optional[List[Dict[str, Any]]] = None,
        order_column: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> str:
        """Register a mutable companion table and return its catalog id.

        `schema` is the table's `pyarrow.Schema`; `primary_key` is a non-empty
        list of column names drawn from it. `indexes` is an optional list of
        ``{"name", "columns", "unique"}`` dicts — one secondary index per entry.
        `order_column` is an optional monotonic column enabling streaming
        `scan_after` reads; `chunk_size` overrides the engine's default scan chunk
        size. Tenant scope is the session's bound tenant. Same handle shape and
        verb signature as the remote `RemoteDatabase.create_mutable_table`; the
        request is assembled with the shared `CreateMutableTableRequest` builder
        and submitted through the engine's wire seam.
        """
        request = build_create_mutable_table_request(
            name,
            schema=schema,
            primary_key=primary_key,
            indexes=indexes,
            order_column=order_column,
            chunk_size=chunk_size,
        )
        return self._native._create_mutable_table_proto(request.SerializeToString())

    def register_topic(
        self,
        name: str,
        *,
        schema: pa.Schema,
        broker_metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a trigger-stream topic and return its engine-minted topic id.

        `schema` is the contract every published batch must satisfy.
        `broker_metadata` is opaque driver-side configuration (retention,
        replication, …). Tenant scope is the session's bound tenant. Same handle
        shape and verb signature as the remote `RemoteDatabase.register_topic`; the
        request is assembled with the shared `RegisterTopicRequest` builder and
        submitted through the engine's wire seam.
        """
        request = build_register_topic_request(
            name,
            schema=schema,
            broker_metadata=broker_metadata,
        )
        return self._native._register_topic_proto(request.SerializeToString())

    def eval_embeddings(
        self,
        *,
        source: str,
        golden_source: str,
        embedding_table: Optional[str] = None,
        k: int = 10,
        cohorts: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate embedding retrieval quality against a golden relevance set.

        Returns a nested dict with ``eval_run_id``, ``aggregate`` (mean
        ``recall_at_k`` / ``precision_at_k`` / ``mrr`` / ``ndcg`` over all
        queries), and ``per_query`` (one record per golden-set query).
        ``embedding_table`` names the result table to evaluate; ``None`` resolves
        the source's most recent embedding table. ``golden_source`` addresses the
        golden set by its full catalog path (``<source>.public.<table>``) or bare
        name. ``cohorts`` optionally maps a golden-set ``query_id`` to an opaque
        ``{key: value}`` segment map, persisted with that query's per-query
        metrics (read back via :meth:`eval_per_query`). Same handle shape and verb
        signature as the remote `RemoteDatabase.eval_embeddings`; the request is
        assembled with the shared `EvalEmbeddingsRequest` builder and submitted
        through the engine's wire seam.
        """
        request = build_eval_embeddings_request(
            source=source,
            golden_source=golden_source,
            embedding_table=embedding_table,
            k=k,
            cohorts=cohorts,
        )
        return self._native._eval_embeddings_proto(request.SerializeToString())

    def eval_per_query(self, eval_run_id: str) -> List[Dict[str, Any]]:
        """Read back the persisted per-query eval records for a run, scoped to
        the calling tenant.

        Returns a list of dicts, each carrying ``eval_run_id``, ``query_id``,
        ``cohorts`` (a dict), and ``metrics`` (a dict of ``recall@1/3/5/10``,
        ``mrr``, ``ndcg``, ``distance``). Same handle shape and verb signature as
        the remote `RemoteDatabase.eval_per_query`; the request is assembled with
        the shared `EvalPerQueryRequest` builder and submitted through the
        engine's wire seam.
        """
        request = build_eval_per_query_request(eval_run_id)
        return self._native._eval_per_query_proto(request.SerializeToString())

    def eval_inference(
        self,
        *,
        model: str,
        source: str,
        columns: List[str],
        task: str,
        golden_source: str,
        label_column: str,
    ) -> Dict[str, Any]:
        """Evaluate inference quality against golden labels.

        Returns a nested dict with ``aggregate`` (task-shaped, tagged by
        ``"task"``) and ``per_record`` (one tagged record per predicted/gold
        pair). ``task`` is ``"classification"`` or ``"ner"``; ``golden_source``
        addresses the golden labels by full catalog path
        (``<source>.public.<table>``) or bare name, with ``label_column`` naming
        the gold-label column. Same handle shape and verb signature as the remote
        `RemoteDatabase.eval_inference`; the request is assembled with the shared
        `EvalInferenceRequest` builder and submitted through the engine's wire
        seam.
        """
        request = build_eval_inference_request(
            model=model,
            source=source,
            columns=columns,
            task=task,
            golden_source=golden_source,
            label_column=label_column,
        )
        return self._native._eval_inference_proto(request.SerializeToString())

    def eval_compare(
        self,
        *,
        embedding_tables: List[str],
        source: str,
        golden_source: str,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Compare multiple embedding tables side-by-side against one golden set.

        Returns a nested dict with a ``per_table`` list whose first entry is the
        baseline (``delta: None``) and whose subsequent entries carry a ``delta``
        against it (per-metric absolute/relative deltas plus paired
        ``significance``, ``None`` when the runs share no query to pair on).
        ``golden_source`` addresses the golden set by full catalog path
        (``<source>.public.<table>``) or bare name. Same handle shape and verb
        signature as the remote `RemoteDatabase.eval_compare`; the request is
        assembled with the shared `EvalCompareRequest` builder and submitted
        through the engine's wire seam.
        """
        request = build_eval_compare_request(
            embedding_tables=embedding_tables,
            source=source,
            golden_source=golden_source,
            k=k,
        )
        return self._native._eval_compare_proto(request.SerializeToString())

    def eval_calibration(
        self,
        *,
        source: str,
        golden_source: str,
        shape: str,
        cohorts: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Evaluate whether a predictor's uncertainty is honest against a held-out
        golden set pairing a predictive distribution with its realised outcome.

        Returns a nested dict with ``aggregate`` (``crps`` / ``nll`` /
        ``adaptive_ece`` / ``sharpness`` / ``coverage`` / ``n``), ``per_cohort``,
        ``per_record``, and ``eval_run_id``. ``shape`` is ``"gaussian"`` or
        ``"sample"``. ``cohorts`` optionally maps a ``record_id`` to an opaque
        ``{key: value}`` segment map persisted with that record's per-record
        scores. Same handle shape and verb signature as the remote
        `RemoteDatabase.eval_calibration`; the request is assembled with the
        shared `EvalCalibrationRequest` builder and submitted through the engine's
        wire seam.
        """
        request = build_eval_calibration_request(
            source=source,
            golden_source=golden_source,
            shape=shape,
            cohorts=cohorts,
        )
        return self._native._eval_calibration_proto(request.SerializeToString())
