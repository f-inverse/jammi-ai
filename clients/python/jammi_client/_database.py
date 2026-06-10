"""`RemoteDatabase` — the transport-agnostic verb surface over a gRPC channel.

This is the pure-Python peer of the Rust `RemoteSession` and the embed wheel's
remote binding: the same verb vocabulary (add-source, generate-embeddings,
encode-query, search, the registry-introspection pair, the tenant trio, the
server handshake), only the transport differs. Every verb delegates 1:1 to a
`jammi.v1` gRPC RPC over one shared channel; encoding/decoding is entirely the
generated stubs' job.

Tenant scope is keyed by an opaque per-connection session id, minted at connect
time and injected as the `jammi-session-id` header on every request — the same
mechanism the Rust SDK and npm client use, so the server's tenant interceptor
binds the same way regardless of which client speaks.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import grpc
import pyarrow as pa
import pyarrow.flight  # noqa: F401  (registers the `pa.flight` submodule)

from . import _conformal
from ._credentials import AnonymousCredentials, ChannelCredentials
from ._errors import TrainingError
from ._generated.jammi.v1 import catalog_pb2, catalog_pb2_grpc
from ._generated.jammi.v1 import embedding_pb2, embedding_pb2_grpc
from ._generated.jammi.v1 import eval_pb2, eval_pb2_grpc
from ._generated.jammi.v1 import inference_pb2, inference_pb2_grpc
from ._generated.jammi.v1 import pipeline_pb2, pipeline_pb2_grpc
from ._generated.jammi.v1 import training_pb2, training_pb2_grpc

# The header carrying a connection's opaque session id. The server's tenant
# interceptor reads it on every request and resolves the bound tenant; it is the
# same key the Rust SDK and npm client use. Kept in one place so the value never
# drifts across the seam.
SESSION_HEADER = "jammi-session-id"

# snake-case modality string → the `Modality` enum the wire carries. One map,
# shared by every `modality=` parameter; the unified form (no per-modality
# method names) mirrors the engine's one `GenerateEmbeddings` / `EncodeQuery`
# verb pair keyed by modality.
_MODALITY = {
    "text": embedding_pb2.Modality.TEXT,
    "image": embedding_pb2.Modality.IMAGE,
    "audio": embedding_pb2.Modality.AUDIO,
}

# Wire `SourceKind` enum value → the snake-case string the embed wheel's
# `SourceType` serialises to, so a descriptor dict is shaped identically whether
# it crossed the gRPC wire or came back from the in-process engine.
_SOURCE_KIND_NAME = {
    catalog_pb2.SourceKind.SOURCE_KIND_FILE: "File",
    catalog_pb2.SourceKind.SOURCE_KIND_POSTGRES: "Postgres",
    catalog_pb2.SourceKind.SOURCE_KIND_MYSQL: "MySql",
}

# File-format string → wire `FileFormat` enum. Mirrors the engine's `FileFormat`
# parse so `add_source(format=...)` accepts the same vocabulary as the embed
# wheel's local path.
_FILE_FORMAT = {
    "parquet": catalog_pb2.FileFormat.FILE_FORMAT_PARQUET,
    "csv": catalog_pb2.FileFormat.FILE_FORMAT_CSV,
    "json": catalog_pb2.FileFormat.FILE_FORMAT_JSON,
    "avro": catalog_pb2.FileFormat.FILE_FORMAT_AVRO,
}


def _modality_value(modality: Optional[str]) -> int:
    """Resolve a `modality=` argument to its wire enum, defaulting to text."""
    if modality is None:
        return embedding_pb2.Modality.TEXT
    try:
        return _MODALITY[modality]
    except KeyError:
        raise ValueError(
            f"modality must be 'text', 'image', or 'audio' (got {modality!r})"
        ) from None


def _local_source_url(url: str) -> str:
    """Normalise a bare local path into a `file://` URL.

    Mirrors the engine's `SourceConnection::parse`, which accepts either a
    storage URL (`file://`, `s3://`, …) or a bare local path and wraps the
    latter as `file://`. A path that already carries a scheme is passed through
    untouched.
    """
    if "://" in url:
        return url
    # A relative path resolves against the SERVER's working dir, matching the
    # embed wheel; we only add the scheme prefix, never resolve client-side.
    return "file://" + url


def _result_table_to_dict(rt: embedding_pb2.ResultTable) -> Dict[str, Any]:
    """Project a wire `ResultTable` into the embed wheel's result-table dict."""
    return {
        "table_name": rt.table_name,
        "source_id": rt.source_id,
        "model_id": rt.model_id,
        "dimensions": rt.dimensions,
        "row_count": rt.row_count,
        "status": rt.status,
    }


def _source_descriptor_to_dict(d: catalog_pb2.SourceDescriptor) -> Dict[str, Any]:
    """Project a wire `SourceDescriptor` into the embed wheel's descriptor dict.

    The descriptor shape matches the embed wheel's `list_sources` /
    `describe_source` entries — `source_id`, `source_type`, `status`, and
    `result_tables` — so a caller reads the same keys regardless of transport.
    """
    return {
        "source_id": d.source_id,
        "source_type": _SOURCE_KIND_NAME.get(d.kind, "Unspecified"),
        "status": d.status,
        "result_tables": [_result_table_to_dict(rt) for rt in d.result_tables],
    }


def _hits_to_table(hits: List[embedding_pb2.SearchHit]) -> pa.Table:
    """Build a `pyarrow.Table` from search hits.

    The columns are `key` + `score` plus one column per projected `select`
    field (stringified on the wire), matching the keyed+scored shape the embed
    wheel's `search` returns.
    """
    keys = [h.key for h in hits]
    scores = [h.score for h in hits]
    columns: Dict[str, List[Any]] = {"key": keys, "score": scores}
    # Projected columns are sparse per-hit on the wire; union the key set so a
    # hit missing a projected column gets a null rather than a ragged table.
    projected: List[str] = []
    for h in hits:
        for col in h.columns:
            if col not in columns:
                columns[col] = [None] * len(hits)
                projected.append(col)
    for i, h in enumerate(hits):
        for col, val in h.columns.items():
            columns[col][i] = val
    return pa.table(columns)


def _arrow_batch_to_table(batch: Any) -> pa.Table:
    """Decode an `ArrowBatch` (one self-describing IPC stream in `data_body`)
    into a `pyarrow.Table`.

    An empty batch (no value columns requested) carries no schema, so it decodes
    to an empty table — matching the embed binding, which returns an empty
    `pyarrow.Table` for an empty `value_rows`.
    """
    body = bytes(batch.data_header) + bytes(batch.data_body)
    if not body:
        return pa.table({})
    reader = pa.ipc.open_stream(body)
    return reader.read_all()


def _calibration_report_to_dict(
    report: eval_pb2.CalibrationEvalReport,
) -> Dict[str, Any]:
    """Project a wire `CalibrationEvalReport` into the embed wheel's nested dict.

    The shape matches the embed `Database.eval_calibration` (which serializes the
    engine's `CalibrationEvalReport` to a dict): ``eval_run_id`` plus an
    ``aggregate`` block, the ``per_cohort`` slices (with optional CI bounds), and
    the ``per_record`` scores — so a caller reads the same keys on both transports.
    """
    aggregate = report.aggregate
    return {
        "eval_run_id": report.eval_run_id,
        "aggregate": {
            "n": aggregate.n,
            "crps": aggregate.crps,
            "nll": aggregate.nll,
            "adaptive_ece": aggregate.adaptive_ece,
            "sharpness": aggregate.sharpness,
            "coverage": aggregate.coverage,
        },
        "per_cohort": [
            {
                "key": c.key,
                "value": c.value,
                "n": c.n,
                "crps": c.crps,
                # Presence-wrapped on the wire: a singleton cohort has no CI, so
                # the embed dict carries `None` there rather than a fabricated 0.
                "crps_ci_lower": c.crps_ci_lower if c.HasField("crps_ci_lower") else None,
                "crps_ci_upper": c.crps_ci_upper if c.HasField("crps_ci_upper") else None,
                "coverage": c.coverage,
            }
            for c in report.per_cohort
        ],
        "per_record": [
            {
                "record_id": r.record_id,
                "crps": r.crps,
                "nll": r.nll,
                "pit": r.pit,
                "covered": r.covered,
                "interval_width": r.interval_width,
                "cohorts": dict(r.cohorts),
            }
            for r in report.per_record
        ],
    }


# Fine-tune method string → the wire `FineTuneMethod` enum. LoRA is the only
# method, matching the embed binding's `method` argument.
_FINE_TUNE_METHOD = {"lora": training_pb2.FineTuneMethod.LORA}

# ModelTask string → the wire `ModelTask` enum, mirroring the embed binding's
# `task=` argument (and the engine's `ModelTask` parse).
_MODEL_TASK = {
    "text_embedding": inference_pb2.ModelTask.TEXT_EMBEDDING,
    "image_embedding": inference_pb2.ModelTask.IMAGE_EMBEDDING,
    "audio_embedding": inference_pb2.ModelTask.AUDIO_EMBEDDING,
    "classification": inference_pb2.ModelTask.CLASSIFICATION,
    "ner": inference_pb2.ModelTask.NER,
    "regression": inference_pb2.ModelTask.REGRESSION,
}

# Backbone-dtype string → the wire `BackboneDtype` enum (the embed binding's
# `backbone_dtype=` vocabulary).
_BACKBONE_DTYPE = {
    "f32": training_pb2.BackboneDtype.F32,
    "bf16": training_pb2.BackboneDtype.BF16,
    "f16": training_pb2.BackboneDtype.F16,
}

# Early-stopping-metric string → the wire enum.
_EARLY_STOPPING_METRIC = {
    "val_loss": training_pb2.EarlyStoppingMetric.VAL_LOSS,
    "train_loss": training_pb2.EarlyStoppingMetric.TRAIN_LOSS,
}

# Context-predictor architecture string → the wire enum.
_CONTEXT_ARCHITECTURE = {
    "cnp": training_pb2.ContextArchitecture.CNP,
    "attncnp": training_pb2.ContextArchitecture.ATTN_CNP,
    "tnp": training_pb2.ContextArchitecture.TNP,
}

# Edge-provenance / edge-direction string → the wire enums (the embed binding's
# `edge_provenance=` / `edge_direction=` vocabularies).
_EDGE_PROVENANCE = {
    "declared": training_pb2.EdgeProvenance.DECLARED,
    "similarity": training_pb2.EdgeProvenance.SIMILARITY,
}
_EDGE_DIRECTION = {
    "out": inference_pb2.EdgeDirection.OUT,
    "in": inference_pb2.EdgeDirection.IN,
    "undirected": inference_pb2.EdgeDirection.UNDIRECTED,
}

# Neighbour-contribution weightings, matching the engine's `PropagationWeighting`.
_PROPAGATION_WEIGHTING = {
    "degree_normalized": pipeline_pb2.PropagationWeighting.PROPAGATION_WEIGHTING_DEGREE_NORMALIZED,
    "uniform": pipeline_pb2.PropagationWeighting.PROPAGATION_WEIGHTING_UNIFORM,
    "edge_similarity": pipeline_pb2.PropagationWeighting.PROPAGATION_WEIGHTING_EDGE_SIMILARITY,
}

# Propagation output modes, matching the engine's `PropagationOutput`.
_PROPAGATION_OUTPUT = {
    "final": pipeline_pb2.PropagationOutput.PROPAGATION_OUTPUT_FINAL,
    "jumping_knowledge": pipeline_pb2.PropagationOutput.PROPAGATION_OUTPUT_JUMPING_KNOWLEDGE,
}

# Context-set pooling reductions, matching the engine's `SetAggregator`.
_SET_AGGREGATOR = {
    "mean": pipeline_pb2.SetAggregator.SET_AGGREGATOR_MEAN,
    "sum": pipeline_pb2.SetAggregator.SET_AGGREGATOR_SUM,
    "max": pipeline_pb2.SetAggregator.SET_AGGREGATOR_MAX,
}

# Calibration predictive shapes, matching the engine's `EvalCalibrationShape`.
_CALIBRATION_SHAPE = {
    "gaussian": eval_pb2.CalibrationShape.CALIBRATION_SHAPE_GAUSSIAN,
    "sample": eval_pb2.CalibrationShape.CALIBRATION_SHAPE_SAMPLE,
}

# Terminal training-job states, matching the engine's `TrainingJobStatus`.
_TERMINAL_STATES = {"completed", "failed"}


def _embedding_loss_message(
    embedding_loss: Optional[str],
    *,
    triplet_margin: Optional[float] = None,
    mnrl_temperature: Optional[float] = None,
) -> Optional[training_pb2.EmbeddingLoss]:
    """Build the wire `EmbeddingLoss` oneof from the embed binding's named loss.

    Mirrors the embed binding's `fine_tune` loss decoding: a named loss with its
    scalar knob (triplet margin / MNRL temperature), or `None` to let the engine
    auto-select from the data format. An unnamed loss with only `triplet_margin`
    set keeps the margin-implies-triplet shorthand.
    """
    if embedding_loss is None:
        if triplet_margin is not None:
            return training_pb2.EmbeddingLoss(
                triplet=training_pb2.EmbeddingLoss.Triplet(margin=triplet_margin)
            )
        return None
    if embedding_loss == "cosent":
        return training_pb2.EmbeddingLoss(co_sent=training_pb2.EmbeddingLoss.CoSent())
    if embedding_loss == "angle":
        return training_pb2.EmbeddingLoss(angle=training_pb2.EmbeddingLoss.AnglE())
    if embedding_loss == "cosine_mse":
        return training_pb2.EmbeddingLoss(
            cosine_mse=training_pb2.EmbeddingLoss.CosineMse()
        )
    if embedding_loss == "triplet":
        return training_pb2.EmbeddingLoss(
            triplet=training_pb2.EmbeddingLoss.Triplet(
                margin=triplet_margin if triplet_margin is not None else 0.3
            )
        )
    if embedding_loss == "mnrl":
        return training_pb2.EmbeddingLoss(
            multiple_negatives_ranking=training_pb2.EmbeddingLoss.MultipleNegativesRanking(
                temperature=mnrl_temperature if mnrl_temperature is not None else 20.0
            )
        )
    raise ValueError(
        f"Unknown embedding_loss {embedding_loss!r}. Use 'cosent', 'angle', "
        f"'cosine_mse', 'triplet', or 'mnrl'."
    )


class RemoteTrainingJob:
    """Handle to a remote training job, polled over `TrainingService`.

    The pure-Python peer of the embed wheel's `TrainingJob`: same handle shape —
    ``job_id`` / ``model_id`` properties, a ``status()`` poll, and a ``wait()``
    that blocks until a terminal state and raises on failure with the wire error.
    `model_id` is the deterministic output id `StartTraining` returned at submit
    time; `wait()` polls `TrainingStatus` and, on ``failed``, raises
    :class:`jammi_client.TrainingError` carrying the worker's error message —
    mirroring the embedded `TrainingJob.wait`.
    """

    # Poll interval for `wait()`. Matches the embedded `TrainingJob.wait`'s
    # 100ms catalog poll, so a remote wait turns around as promptly.
    _POLL_INTERVAL_SECONDS = 0.1

    def __init__(
        self,
        stub: training_pb2_grpc.TrainingServiceStub,
        metadata,
        *,
        job_id: str,
        model_id: str,
    ) -> None:
        self._stub = stub
        self._metadata = metadata
        self._job_id = job_id
        self._model_id = model_id

    @property
    def job_id(self) -> str:
        """The unique job id `StartTraining` assigned."""
        return self._job_id

    @property
    def model_id(self) -> str:
        """The deterministic output model id the trained artifact registers under."""
        return self._model_id

    def _status_response(self) -> training_pb2.TrainingStatusResponse:
        return self._stub.TrainingStatus(
            training_pb2.TrainingStatusRequest(job_id=self._job_id),
            metadata=self._metadata,
        )

    def status(self) -> str:
        """The job's current status string. Maps to `TrainingService.TrainingStatus`."""
        return self._status_response().status

    def wait(self) -> None:
        """Block until the job reaches a terminal state; raise on failure.

        Polls `TrainingStatus` until ``completed`` (returns) or ``failed`` (raises
        :class:`jammi_client.TrainingError` with the wire error message) — the
        remote peer of the embedded `TrainingJob.wait`.
        """
        while True:
            resp = self._status_response()
            if resp.status == "completed":
                return
            if resp.status == "failed":
                raise TrainingError(resp.error or "training job failed")
            time.sleep(self._POLL_INTERVAL_SECONDS)


class RemoteDatabase:
    """A Database driving a remote jammi engine over the `jammi.v1` gRPC wire.

    The verbs are the transport-agnostic surface; an embedded `Database` (in the
    `jammi-ai` wheel) and a `RemoteDatabase` expose the same vocabulary, only the
    transport differs. Use as a context manager to close the channel on exit.
    """

    def __init__(
        self,
        channel: grpc.Channel,
        *,
        session_id: str,
        endpoint: str,
        tls: bool,
    ) -> None:
        self._channel = channel
        self._session_id = session_id
        self._endpoint = endpoint
        self._tls = tls
        self._metadata = ((SESSION_HEADER, session_id),)
        self._embedding = embedding_pb2_grpc.EmbeddingServiceStub(channel)
        self._catalog = catalog_pb2_grpc.CatalogServiceStub(channel)
        self._training = training_pb2_grpc.TrainingServiceStub(channel)
        self._inference = inference_pb2_grpc.InferenceServiceStub(channel)
        self._pipeline = pipeline_pb2_grpc.PipelineServiceStub(channel)
        self._eval = eval_pb2_grpc.EvalServiceStub(channel)
        # The Flight SQL client shares the gRPC endpoint (the server co-mounts
        # Flight SQL and the typed gRPC services on one Tonic port); built lazily
        # on first `sql` so a connection that never queries SQL pays nothing.
        self._flight = None

    @property
    def session_id(self) -> str:
        """The opaque session id the server keys this connection's tenant by."""
        return self._session_id

    # --- Catalog: tenant trio + handshake ---------------------------------------

    def with_tenant(self, tenant_id: str) -> None:
        """Bind a tenant scope to this connection. Pass an empty string to clear.

        Maps to `CatalogService.SetTenant` / `ClearTenant`, keyed by this
        connection's session id.
        """
        if tenant_id == "":
            self._catalog.ClearTenant(
                _empty(), metadata=self._metadata
            )
            return
        self._catalog.SetTenant(
            catalog_pb2.SetTenantRequest(tenant=catalog_pb2.Tenant(id=tenant_id)),
            metadata=self._metadata,
        )

    def tenant(self) -> Optional[str]:
        """The tenant currently bound to this connection, or ``None``.

        Maps to `CatalogService.GetTenant`.
        """
        resp = self._catalog.GetTenant(_empty(), metadata=self._metadata)
        return resp.tenant.id or None

    def get_server_info(self) -> Dict[str, Any]:
        """The engine's capabilities handshake: ``version`` / ``features`` /
        ``storage_backends`` / ``services``. Maps to
        `CatalogService.GetServerInfo`.

        The first three fields are compile-time facts about the build;
        ``services`` is the runtime tier handshake — the gRPC service tiers this
        deployment mounted (``"core"`` is always present; ``"train"`` /
        ``"event"`` / ``"eval"`` appear only when this server enabled them). A
        client reads ``services`` to know which verbs are reachable here before
        calling them.

        The same keys the embedded `Database.get_server_info` returns, so the
        handshake shape agrees across transports.
        """
        resp = self._catalog.GetServerInfo(_empty(), metadata=self._metadata)
        return {
            "version": resp.version,
            "features": list(resp.features),
            "storage_backends": list(resp.storage_backends),
            "services": list(resp.services),
        }

    # --- Sources -----------------------------------------------------------------

    def add_source(self, name: str, *, url: str, format: str) -> None:
        """Register a file-shaped data source on the remote engine.

        `url` accepts a local path (wrapped into `file://...` server-side) or any
        storage URL the server was compiled with (`s3://`, `gs://`, `azure://`).
        Maps to `CatalogService.AddSource`.
        """
        try:
            file_format = _FILE_FORMAT[format]
        except KeyError:
            raise ValueError(
                f"format must be one of {sorted(_FILE_FORMAT)} (got {format!r})"
            ) from None
        self._catalog.AddSource(
            catalog_pb2.AddSourceRequest(
                source_id=name,
                source_kind=catalog_pb2.SourceKind.SOURCE_KIND_FILE,
                connection=catalog_pb2.SourceConnection(
                    url=_local_source_url(url),
                    format=file_format,
                ),
            ),
            metadata=self._metadata,
        )

    def list_sources(self) -> List[Dict[str, Any]]:
        """A descriptor for every source registered to the current tenant.

        Maps to `CatalogService.ListSources`; same dict shape per entry as
        :meth:`describe_source`.
        """
        resp = self._catalog.ListSources(
            catalog_pb2.ListSourcesRequest(), metadata=self._metadata
        )
        return [_source_descriptor_to_dict(d) for d in resp.sources]

    def describe_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Describe one registered source by id, or ``None`` if not visible.

        Maps to `CatalogService.DescribeSource`. The engine returns a NotFound
        status when no such source exists; that is surfaced here as ``None``.
        """
        try:
            d = self._catalog.DescribeSource(
                catalog_pb2.DescribeSourceRequest(source_id=source_id),
                metadata=self._metadata,
            )
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.NOT_FOUND:
                return None
            raise
        return _source_descriptor_to_dict(d)

    # --- Embeddings + search -----------------------------------------------------

    def encode_query(
        self,
        *,
        model: str,
        query: Union[str, bytes],
        modality: Optional[str] = None,
    ) -> List[float]:
        """Encode a single query into an embedding vector with the given model.

        `query` is a string for the text tower or raw bytes for the image/audio
        tower; `modality` selects the tower. Maps to `EmbeddingService.EncodeQuery`.
        """
        request = embedding_pb2.EncodeQueryRequest(
            model_id=model,
            modality=_modality_value(modality),
        )
        if isinstance(query, str):
            request.text = query
        elif isinstance(query, (bytes, bytearray)):
            request.data = bytes(query)
        else:
            raise TypeError(
                "query must be a str (text tower) or bytes (image/audio tower)"
            )
        resp = self._embedding.EncodeQuery(request, metadata=self._metadata)
        return list(resp.embedding)

    def generate_embeddings(
        self,
        *,
        source: str,
        model: str,
        columns: List[str],
        key: str,
        modality: Optional[str] = None,
    ) -> str:
        """Embed `columns` of a registered source, persisting one vector per row.

        `modality` selects the tower. Returns the result table name. Maps to
        `EmbeddingService.GenerateEmbeddings`.
        """
        resp = self._embedding.GenerateEmbeddings(
            embedding_pb2.GenerateEmbeddingsRequest(
                source_id=source,
                model_id=model,
                columns=list(columns),
                key_column=key,
                modality=_modality_value(modality),
            ),
            metadata=self._metadata,
        )
        return resp.table_name

    def search(
        self,
        source: str,
        *,
        query: List[float],
        k: int,
        filter: Optional[str] = None,
        select: Optional[List[str]] = None,
    ) -> pa.Table:
        """Nearest-neighbor search over a source's embedding table.

        `query` is the query vector; `filter` is an optional SQL predicate over
        the hydrated results; `select` projects columns (empty keeps the
        keyed+scored shape). Returns a `pyarrow.Table`. Maps to
        `EmbeddingService.Search`.
        """
        request = embedding_pb2.SearchRequest(
            source_id=source,
            query_vector=embedding_pb2.QueryVector(values=list(query)),
            k=k,
            select=list(select or []),
        )
        if filter is not None:
            request.filter = filter
        resp = self._embedding.Search(request, metadata=self._metadata)
        return _hits_to_table(list(resp.hits))

    # --- Training (offloaded to the remote train tier) ---------------------------
    #
    # These verbs DO hit the wire: training runs on the remote GPU server, so the
    # client submits a spec via `TrainingService.StartTraining` and returns a
    # `RemoteTrainingJob` to poll. The signatures mirror the embed `Database`'s
    # so a caller swaps transports without changing the call. `predict` rides
    # `InferenceService.Predict`.

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
    ) -> RemoteTrainingJob:
        """Submit a LoRA fine-tuning job to the remote engine; poll the handle.

        Returns a :class:`RemoteTrainingJob` — same handle shape and verb
        signature as the embed `Database.fine_tune`. Maps to
        `TrainingService.StartTraining` with the `FineTuneSpec` arm; all config
        kwargs are optional, applying the engine defaults when omitted.
        """
        try:
            wire_method = _FINE_TUNE_METHOD[method]
        except KeyError:
            raise ValueError(
                f"method must be one of {sorted(_FINE_TUNE_METHOD)} (got {method!r})"
            ) from None
        wire_task = _MODEL_TASK[task] if task is not None else _MODEL_TASK["text_embedding"]

        config = self._fine_tune_config(
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
        )
        request = training_pb2.StartTrainingRequest(
            fine_tune=training_pb2.FineTuneSpec(
                source=source,
                columns=list(columns),
                method=wire_method,
                task=wire_task,
            ),
            base_model=base_model,
            config=config,
        )
        return self._start_training(request)

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
    ) -> RemoteTrainingJob:
        """Submit a graph-supervised fine-tune (S11) to the remote engine.

        Returns a :class:`RemoteTrainingJob`, mirroring the embed
        `Database.fine_tune_graph`. Maps to `TrainingService.StartTraining` with
        the `GraphFineTuneSpec` arm. `edge_provenance` is the load-bearing
        circularity distinction — "declared" external edges teach the metric
        something new; "similarity" edges are a weak bootstrap only.
        """
        try:
            provenance = _EDGE_PROVENANCE[edge_provenance]
        except KeyError:
            raise ValueError(
                f"edge_provenance must be 'declared' or 'similarity' "
                f"(got {edge_provenance!r})"
            ) from None

        # The graph defaults match the engine's `GraphSampleConfig::default()`;
        # an omitted knob keeps the engine default.
        sample = training_pb2.GraphSampleConfig(
            walk_length=walk_length if walk_length is not None else 4,
            walks_per_node=walks_per_node if walks_per_node is not None else 2,
            return_p=return_p if return_p is not None else 1.0,
            in_out_q=in_out_q if in_out_q is not None else 1.0,
            hard_negatives=graph_hard_negatives if graph_hard_negatives is not None else 1,
            exclude_hops=exclude_hops if exclude_hops is not None else 1,
            min_negatives=min_negatives if min_negatives is not None else 1,
            seed=sample_seed if sample_seed is not None else 0,
        )

        # The default graph embedding loss is MNRL (S10), matching the embed
        # binding; only 'mnrl' / 'triplet' are accepted for graph supervision.
        if embedding_loss in (None, "mnrl"):
            loss = training_pb2.EmbeddingLoss(
                multiple_negatives_ranking=training_pb2.EmbeddingLoss.MultipleNegativesRanking(
                    temperature=mnrl_temperature if mnrl_temperature is not None else 20.0
                )
            )
        elif embedding_loss == "triplet":
            loss = training_pb2.EmbeddingLoss(
                triplet=training_pb2.EmbeddingLoss.Triplet(margin=0.3)
            )
        else:
            raise ValueError(
                f"Unknown embedding_loss {embedding_loss!r} for graph fine-tune. "
                f"Use 'mnrl' (default) or 'triplet'."
            )
        config = training_pb2.FineTuneConfig(embedding_loss=loss)
        if epochs is not None:
            config.epochs = epochs
        if batch_size is not None:
            config.batch_size = batch_size
        if learning_rate is not None:
            config.learning_rate = learning_rate
        if lora_rank is not None:
            config.lora_rank = lora_rank
        if matryoshka_dims is not None:
            config.matryoshka_dims.extend(matryoshka_dims)

        request = training_pb2.StartTrainingRequest(
            graph_fine_tune=training_pb2.GraphFineTuneSpec(
                sources=training_pb2.GraphFineTuneSources(
                    node_source=node_source,
                    id_column=id_column,
                    text_column=text_column,
                    edge_source=edge_source,
                    src_column=src_column,
                    dst_column=dst_column,
                    provenance=provenance,
                ),
                sample_config=sample,
            ),
            base_model=base_model,
        )
        return self._start_training(request)

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
    ) -> RemoteTrainingJob:
        """Submit an amortized in-context predictor (S19) meta-training to the
        remote engine.

        Returns a :class:`RemoteTrainingJob`, mirroring the embed
        `Database.train_context_predictor`. Maps to `TrainingService.StartTraining`
        with the `ContextPredictorSpec` arm.
        """
        try:
            wire_architecture = _CONTEXT_ARCHITECTURE[architecture]
        except KeyError:
            raise ValueError(
                f"Unknown architecture {architecture!r}. Use 'cnp', 'attncnp', "
                f"or 'tnp'."
            ) from None

        if output == "gaussian":
            if objective == "crps":
                gaussian = training_pb2.PredictiveHead.Gaussian(
                    objective=training_pb2.GaussianObjective(
                        crps=training_pb2.GaussianObjective.Crps()
                    )
                )
            elif objective == "nll":
                gaussian = training_pb2.PredictiveHead.Gaussian(
                    objective=training_pb2.GaussianObjective(
                        nll=training_pb2.GaussianObjective.Nll(beta=0.0)
                    )
                )
            elif objective == "betanll":
                gaussian = training_pb2.PredictiveHead.Gaussian(
                    objective=training_pb2.GaussianObjective(
                        nll=training_pb2.GaussianObjective.Nll(beta=beta)
                    )
                )
            else:
                raise ValueError(
                    f"Unknown gaussian objective {objective!r}. Use 'crps', 'nll', "
                    f"or 'betanll'."
                )
            head = training_pb2.PredictiveHead(gaussian=gaussian)
        elif output == "quantile":
            if levels is None:
                raise ValueError(
                    "output='quantile' requires `levels` (ascending levels in (0, 1))"
                )
            head = training_pb2.PredictiveHead(
                quantile=training_pb2.PredictiveHead.Quantile(levels=list(levels))
            )
        else:
            raise ValueError(
                f"Unknown output {output!r}. Use 'gaussian' or 'quantile'."
            )

        spec = training_pb2.ContextPredictorTrainConfig(
            model_id=model_id if model_id is not None else f"{source}-context-predictor",
            architecture=wire_architecture,
            key_column=key_column,
            task_column=task_column,
            value_column=value_column,
            context_k=context_k,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            head=head,
            epochs=epochs,
            learning_rate=learning_rate,
            grad_clip=grad_clip,
            test_task_fraction=test_task_fraction,
            min_task_count=min_task_count,
            seed=seed,
        )
        request = training_pb2.StartTrainingRequest(
            context_predictor=training_pb2.ContextPredictorSpec(
                source=source,
                predictor_spec=spec,
            )
        )
        return self._start_training(request)

    def predict_with_context_predictor(
        self,
        model_id: str,
        *,
        source: str,
        target_key: str,
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
        hybrid_ann_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Predict a target's distribution with a trained context predictor (S19).

        Returns the same dict shape as the embed `Database`:
        ``{"kind": "gaussian", "mean", "std", "source", "context_ref"}`` or
        ``{"kind": "quantile", "levels": [[level, value], …], "source",
        "context_ref"}``. Maps to `InferenceService.Predict`. Passing
        `edge_source` assembles a declared-edge context (or, with `hybrid_ann_k`,
        the union with the ANN context) instead of embedding-similarity context.
        """
        request = inference_pb2.PredictRequest(
            model_id=model_id,
            source=source,
            target_key=target_key,
        )
        if split is not None:
            request.split = split
        if edge_source is not None:
            gather = inference_pb2.EdgeGather(
                edge_source=edge_source,
                src_column=edge_src_column if edge_src_column is not None else "src",
                dst_column=edge_dst_column if edge_dst_column is not None else "dst",
            )
            if edge_type_column is not None:
                gather.type_column = edge_type_column
            if edge_weight_column is not None:
                gather.weight_column = edge_weight_column
            if edge_hops is not None:
                gather.hops = edge_hops
            if edge_fanout is not None:
                gather.fanout = edge_fanout
            if edge_direction is not None:
                try:
                    gather.direction = _EDGE_DIRECTION[edge_direction]
                except KeyError:
                    raise ValueError(
                        f"edge_direction must be 'out', 'in', or 'undirected' "
                        f"(got {edge_direction!r})"
                    ) from None
            if edge_types is not None:
                gather.edge_types.extend(edge_types)
            if min_weight is not None:
                gather.min_weight = min_weight
            request.edges.CopyFrom(gather)
        if hybrid_ann_k is not None:
            request.hybrid_ann_k = hybrid_ann_k

        resp = self._inference.Predict(request, metadata=self._metadata)
        out: Dict[str, Any] = {}
        kind = resp.WhichOneof("distribution")
        if kind == "gaussian":
            out["kind"] = "gaussian"
            out["mean"] = resp.gaussian.mean
            out["std"] = resp.gaussian.std
        elif kind == "quantile":
            out["kind"] = "quantile"
            out["levels"] = [[p.level, p.value] for p in resp.quantile.points]
        else:
            raise RuntimeError("Predict response carried no distribution")
        out["source"] = resp.source
        out["context_ref"] = list(resp.context_ref)
        return out

    # --- Engine-state pipeline verbs (PipelineService / EvalService) -------------
    #
    # These build durable graph/embedding artifacts or assemble a target's
    # conditioning context on the remote engine. The two graph-build verbs return
    # the materialised table name (the caller reads it via `sql(...)`); the
    # compute stays server-side, so the table is byte-identical to one built in
    # the embedded engine. `assemble_context` returns the pooled vector inline,
    # and `eval_calibration` the typed calibration report. The signatures mirror
    # the embed `Database`'s so a caller swaps transports without changing the
    # call.

    def build_neighbor_graph(
        self,
        source: str,
        *,
        k: int,
        min_similarity: Optional[float] = None,
        mutual: bool = False,
        exact: bool = False,
        table: Optional[str] = None,
    ) -> str:
        """Materialise the k-NN graph of a source's embedding table and return the
        new edge table's name.

        The returned table has columns ``(src, dst, rank, similarity)``. The
        default driver is index-assisted and approximate; pass ``exact=True`` for
        a deterministic, complete graph. ``min_similarity`` floors weak edges;
        ``mutual=True`` keeps only reciprocal edges. Maps to
        `PipelineService.BuildNeighborGraph`; read the table via :meth:`sql`.
        """
        request = pipeline_pb2.BuildNeighborGraphRequest(
            source_id=source,
            k=k,
            mutual=mutual,
            exact=exact,
        )
        if min_similarity is not None:
            request.min_similarity = min_similarity
        if table is not None:
            request.table = table
        resp = self._pipeline.BuildNeighborGraph(request, metadata=self._metadata)
        return resp.table_name

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
    ) -> str:
        """Propagate an embedding table's features over a declared graph (the
        decoupled-GNN forward pass) into a new, searchable embedding table.

        The graph is either an S9 similarity graph (``edge_graph_table``, a
        :meth:`build_neighbor_graph` output) or a registered external edge source
        (``edge_source``) — pass exactly one. ``weighting`` selects the neighbour
        normalisation; ``output`` is ``"final"`` or ``"jumping_knowledge"``.
        Returns the materialised table's name. Maps to
        `PipelineService.PropagateEmbeddings`; read the table via :meth:`sql`.
        """
        if edge_graph_table is not None and edge_source is not None:
            raise ValueError(
                "pass exactly one of edge_graph_table (S9 graph) or edge_source "
                "(registered edges), not both"
            )
        request = pipeline_pb2.PropagateEmbeddingsRequest(source_id=source)
        if edge_graph_table is not None:
            request.edge_graph_table = edge_graph_table
        elif edge_source is not None:
            request.edge_source.CopyFrom(
                pipeline_pb2.PropagateEdgeSource(
                    edge_source=edge_source,
                    src_column=edge_src_column if edge_src_column is not None else "src",
                    dst_column=edge_dst_column if edge_dst_column is not None else "dst",
                )
            )
            if edge_weight_column is not None:
                request.edge_source.weight_column = edge_weight_column
        else:
            raise ValueError(
                "propagate_embeddings requires a graph: edge_graph_table or edge_source"
            )
        if embedding_table is not None:
            request.embedding_table = embedding_table
        if direction is not None:
            try:
                request.direction = _EDGE_DIRECTION[direction]
            except KeyError:
                raise ValueError(
                    f"direction must be 'out', 'in', or 'undirected' (got {direction!r})"
                ) from None
        if hops is not None:
            request.hops = hops
        if weighting is not None:
            try:
                request.weighting = _PROPAGATION_WEIGHTING[weighting]
            except KeyError:
                raise ValueError(
                    "weighting must be 'degree_normalized', 'uniform', or "
                    f"'edge_similarity' (got {weighting!r})"
                ) from None
        if alpha is not None:
            request.alpha = alpha
        if output is not None:
            try:
                request.output = _PROPAGATION_OUTPUT[output]
            except KeyError:
                raise ValueError(
                    f"output must be 'final' or 'jumping_knowledge' (got {output!r})"
                ) from None
        resp = self._pipeline.PropagateEmbeddings(request, metadata=self._metadata)
        return resp.table_name

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

        Returns the same dict shape as the embed `Database`: ``context_vector``
        (list of floats, or ``None`` for a degenerate empty context),
        ``context_size``, ``context_keys``, ``value_rows`` (a ``pyarrow.Table``),
        and ``source`` (the assembly fact). Maps to
        `PipelineService.AssembleContext`.
        """
        request = pipeline_pb2.AssembleContextRequest(
            source_id=source,
            query=list(query),
            k=k,
            value_columns=list(value_columns or []),
            exclude_self=exclude_self,
            hybrid=hybrid,
        )
        if aggregator is not None:
            try:
                request.aggregator = _SET_AGGREGATOR[aggregator]
            except KeyError:
                raise ValueError(
                    f"aggregator must be 'mean', 'sum', or 'max' (got {aggregator!r})"
                ) from None
        if exclude_key is not None:
            request.exclude_key = exclude_key
        if split is not None:
            request.split = split
        if edge_source is not None:
            gather = inference_pb2.EdgeGather(
                edge_source=edge_source,
                src_column=edge_src_column if edge_src_column is not None else "src",
                dst_column=edge_dst_column if edge_dst_column is not None else "dst",
            )
            if edge_type_column is not None:
                gather.type_column = edge_type_column
            if edge_weight_column is not None:
                gather.weight_column = edge_weight_column
            if edge_hops is not None:
                gather.hops = edge_hops
            if edge_fanout is not None:
                gather.fanout = edge_fanout
            if edge_direction is not None:
                try:
                    gather.direction = _EDGE_DIRECTION[edge_direction]
                except KeyError:
                    raise ValueError(
                        f"edge_direction must be 'out', 'in', or 'undirected' "
                        f"(got {edge_direction!r})"
                    ) from None
            if edge_types is not None:
                gather.edge_types.extend(edge_types)
            if min_weight is not None:
                gather.min_weight = min_weight
            request.edges.CopyFrom(gather)

        resp = self._pipeline.AssembleContext(request, metadata=self._metadata)
        # The pooled vector is presence-wrapped so a degenerate empty context
        # (`None`) stays distinguishable from a present-but-empty vector — the
        # same correctness signal the embed binding's `None` carries.
        context_vector = (
            list(resp.context_vector.values)
            if resp.HasField("context_vector")
            else None
        )
        value_rows = _arrow_batch_to_table(resp.value_rows)
        return {
            "context_vector": context_vector,
            "context_size": resp.context_size,
            "context_keys": list(resp.context_keys),
            "value_rows": value_rows,
            "source": resp.source,
        }

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

        Returns the same nested dict the embed `Database` produces: ``aggregate``
        (``crps`` / ``nll`` / ``adaptive_ece`` / ``sharpness`` / ``coverage`` /
        ``n``), ``per_cohort``, ``per_record``, and ``eval_run_id``. ``shape`` is
        ``"gaussian"`` or ``"sample"``. Maps to `EvalService.EvalCalibration`.
        """
        try:
            shape_value = _CALIBRATION_SHAPE[shape]
        except KeyError:
            raise ValueError(
                f"shape must be 'gaussian' or 'sample' (got {shape!r})"
            ) from None
        request = eval_pb2.EvalCalibrationRequest(
            source_id=source,
            golden_source=golden_source,
            shape=shape_value,
        )
        for record_id, tags in (cohorts or {}).items():
            request.cohorts[record_id].tags.update(tags)
        resp = self._eval.EvalCalibration(request, metadata=self._metadata)
        return _calibration_report_to_dict(resp)

    def _start_training(
        self, request: training_pb2.StartTrainingRequest
    ) -> RemoteTrainingJob:
        """Submit a `StartTraining` request and wrap the response in a handle."""
        resp = self._training.StartTraining(request, metadata=self._metadata)
        return RemoteTrainingJob(
            self._training,
            self._metadata,
            job_id=resp.job_id,
            model_id=resp.model_id,
        )

    def _fine_tune_config(
        self,
        *,
        lora_rank: Optional[int],
        lora_alpha: Optional[float],
        lora_dropout: Optional[float],
        learning_rate: Optional[float],
        epochs: Optional[int],
        batch_size: Optional[int],
        max_seq_length: Optional[int],
        validation_fraction: Optional[float],
        early_stopping_patience: Optional[int],
        warmup_steps: Optional[int],
        gradient_accumulation_steps: Optional[int],
        triplet_margin: Optional[float],
        target_modules: Optional[List[str]],
        early_stopping_metric: Optional[str],
        backbone_dtype: Optional[str],
        weight_decay: Optional[float],
        max_grad_norm: Optional[float],
        embedding_loss: Optional[str],
        mnrl_temperature: Optional[float],
        cached: Optional[bool],
        mine_hard_negatives: Optional[bool],
        hard_negative_k: Optional[int],
        hard_negative_exclude_hops: Optional[int],
        hard_negative_refresh_every: Optional[int],
        matryoshka_dims: Optional[List[int]],
    ) -> training_pb2.FineTuneConfig:
        """Build the wire `FineTuneConfig` from the embed binding's flat kwargs.

        Every scalar knob on the proto has explicit presence (`optional`), so an
        omitted kwarg leaves its field UNSET on the wire — not stamped to a proto
        default `0`. The server decode starts from `FineTuneConfig::default()` and
        overrides a field only when it is present, so an unset field resolves to
        the engine default (never a literal zero). Only the fields a caller
        actually set are sent; the engine fills the rest. The engine is the single
        source of default values, so this client duplicates none of them.
        """
        config = training_pb2.FineTuneConfig()
        loss = _embedding_loss_message(
            embedding_loss,
            triplet_margin=triplet_margin,
            mnrl_temperature=mnrl_temperature,
        )
        if loss is not None:
            config.embedding_loss.CopyFrom(loss)
        if lora_rank is not None:
            config.lora_rank = lora_rank
        if lora_alpha is not None:
            config.lora_alpha = lora_alpha
        if lora_dropout is not None:
            config.lora_dropout = lora_dropout
        if learning_rate is not None:
            config.learning_rate = learning_rate
        if epochs is not None:
            config.epochs = epochs
        if batch_size is not None:
            config.batch_size = batch_size
        if max_seq_length is not None:
            config.max_seq_length = max_seq_length
        if validation_fraction is not None:
            config.validation_fraction = validation_fraction
        if early_stopping_patience is not None:
            config.early_stopping_patience = early_stopping_patience
        if warmup_steps is not None:
            config.warmup_steps = warmup_steps
        if gradient_accumulation_steps is not None:
            config.gradient_accumulation_steps = gradient_accumulation_steps
        if target_modules is not None:
            config.target_modules.extend(target_modules)
        if early_stopping_metric is not None:
            try:
                config.early_stopping_metric = _EARLY_STOPPING_METRIC[early_stopping_metric]
            except KeyError:
                raise ValueError(
                    f"Unknown early_stopping_metric {early_stopping_metric!r}. "
                    f"Use 'val_loss' or 'train_loss'."
                ) from None
        if backbone_dtype is not None:
            try:
                config.backbone_dtype = _BACKBONE_DTYPE[backbone_dtype]
            except KeyError:
                raise ValueError(
                    f"Unknown backbone_dtype {backbone_dtype!r}. Use 'f32', "
                    f"'bf16', or 'f16'."
                ) from None
        if weight_decay is not None:
            config.weight_decay = weight_decay
        if max_grad_norm is not None:
            config.max_grad_norm = max_grad_norm
        if cached is not None:
            config.cached = cached
        # The hard-negative mining knobs share one nested message; stamp it only
        # when a caller set any of them, matching the embed binding's mutation of
        # the default `HardNegativeConfig`.
        if any(
            v is not None
            for v in (
                mine_hard_negatives,
                hard_negative_k,
                hard_negative_exclude_hops,
                hard_negative_refresh_every,
            )
        ):
            hn = training_pb2.HardNegativeConfig()
            if mine_hard_negatives is not None:
                hn.mine = mine_hard_negatives
            if hard_negative_k is not None:
                hn.k = hard_negative_k
            if hard_negative_exclude_hops is not None:
                hn.exclude_hops = hard_negative_exclude_hops
            if hard_negative_refresh_every is not None:
                hn.refresh_every = hard_negative_refresh_every
            config.hard_negatives.CopyFrom(hn)
        if matryoshka_dims is not None:
            config.matryoshka_dims.extend(matryoshka_dims)
        return config

    # --- Stateless conformal / RRF numerics (computed client-side) ---------------
    #
    # These four verbs are pure functions of caller-supplied arrays — the server
    # holds none of their inputs, so they make NO gRPC hop. They run locally,
    # delegating to `_conformal`, which reproduces the embedded engine's
    # algorithm exactly so the verb surface agrees on both transports. Their
    # signatures match the embedded `Database`'s verbs of the same names.

    def conformalize(
        self,
        calibration: Sequence[Sequence[float]],
        true_labels: Sequence[int],
        test: Sequence[Sequence[float]],
        *,
        alpha: float,
        score: Optional[str] = None,
        raps_params: Optional[Tuple[float, int]] = None,
    ) -> List[List[int]]:
        """Conformalize a classification predictor into prediction sets.

        Computed **client-side**: the calibration probabilities, ``true_labels``,
        and ``test`` rows are caller-supplied arrays the engine never holds, so a
        wire hop would only ship data the caller already has. The
        finite-sample ``⌈(n+1)(1-alpha)⌉`` quantile over the calibration scores
        is applied to every ``test`` row to emit a prediction set with marginal
        coverage ``>= 1 - alpha``.

        ``score`` selects the nonconformity family — ``"lac"``, ``"aps"``
        (default), or ``"raps"``; for ``"raps"``, ``raps_params`` is the
        ``(lambda, k_reg)`` pair (default ``(0.0, 1)``). Same algorithm and
        output as the embedded engine's ``conformalize``.
        """
        return _conformal.conformalize(
            calibration,
            true_labels,
            test,
            alpha=alpha,
            score=score,
            raps_params=raps_params,
        )

    def conformalize_interval(
        self,
        predictions: Sequence[float],
        observed: Sequence[float],
        test_predictions: Sequence[float],
        *,
        alpha: float,
    ) -> List[Tuple[float, float]]:
        """Conformalize an absolute-residual regression predictor into intervals.

        Computed **client-side**: ``predictions``, ``observed``, and
        ``test_predictions`` are caller-supplied arrays not held by the engine,
        so no gRPC hop is made. The calibration nonconformity ``|y - ŷ|`` yields
        the finite-sample quantile ``q̂``, giving ``[ŷ - q̂, ŷ + q̂]`` per test
        point. Same algorithm and output as the embedded engine's
        ``conformalize_interval``.
        """
        return _conformal.conformalize_interval(
            predictions, observed, test_predictions, alpha=alpha
        )

    def conformalize_cqr(
        self,
        lower: Sequence[float],
        upper: Sequence[float],
        observed: Sequence[float],
        test_lower: Sequence[float],
        test_upper: Sequence[float],
        *,
        alpha: float,
    ) -> List[Tuple[float, float]]:
        """Conformalize a Conformalized Quantile Regression predictor into intervals.

        Computed **client-side**: the lower/upper quantile bands, ``observed``
        targets, and test bands are caller-supplied arrays the engine never
        holds, so no wire hop is made. The calibration nonconformity
        ``max(q_lo - y, y - q_hi)`` yields the finite-sample quantile ``q̂``,
        giving the adaptive-width ``[q_lo - q̂, q_hi + q̂]`` per test row. Same
        algorithm and output as the embedded engine's ``conformalize_cqr``.
        """
        return _conformal.conformalize_cqr(
            lower, upper, observed, test_lower, test_upper, alpha=alpha
        )

    def rrf_fuse(
        self,
        ranked_lists: Sequence[Sequence[str]],
        *,
        k_rrf: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Fuse several ranked retrieval lists by reciprocal-rank fusion.

        Computed **client-side**: ``ranked_lists`` is caller-supplied — the
        already-ranked id lists are in hand, so fusing them needs no engine
        state and makes no gRPC hop. A row's fused score is
        ``Σ 1 / (k_rrf + rank + 1)``; the result is sorted by score descending,
        ties ascending by ``row_id``. ``k_rrf`` defaults to 60. Same algorithm
        and output as the embedded engine's ``rrf_fuse``.
        """
        return _conformal.rrf_fuse(ranked_lists, k_rrf=k_rrf)

    # --- Compound query (Flight SQL) --------------------------------------------

    def sql(self, query: str) -> pa.Table:
        """Run a SQL query against the remote engine over the Flight SQL lane.

        This is the open, caller-shaped compound surface — `join` / `filter` /
        `select` over registered sources, and crucially `annotate(...)`, the
        engine's model-inference table function, so compound retrieval +
        inference run in one round-trip:

        ```python
        db.sql(
            "SELECT a._row_id, a.vector "
            "FROM annotate('local:/models/bert', 'text_embedding', "
            "              'docs.public.papers', 'id', 'abstract') AS a"
        )
        ```

        The same `jammi-session-id` that scopes this connection's tenant on the
        typed gRPC verbs is sent on the Flight SQL query, so SQL reads observe
        the same tenant scope. Returns a `pyarrow.Table`. The embedded
        `Database.sql` is the in-process peer of this verb — same SQL, same
        `annotate` function, transport apart.
        """
        client = self._flight_client()
        options = self._flight_options()
        info = client.get_flight_info(
            pa.flight.FlightDescriptor.for_command(_command_statement_query(query)),
            options,
        )
        reader = client.do_get(info.endpoints[0].ticket, options)
        return reader.read_all()

    def _flight_client(self):
        # The Flight SQL lane is a separate pyarrow.flight transport that does
        # not flow through the gRPC channel credentials, so a channel-level
        # bearer does not reach it yet; tracked at
        # https://github.com/f-inverse/jammi-ai/issues/96. It still carries the
        # session header below.
        if self._flight is None:
            scheme = "grpc+tls" if self._tls else "grpc+tcp"
            self._flight = pa.flight.FlightClient(f"{scheme}://{self._endpoint}")
        return self._flight

    def _flight_options(self):
        # Carry the connection's session id as the tenant-scoping header, the
        # same key the typed gRPC verbs send.
        return pa.flight.FlightCallOptions(
            headers=[(SESSION_HEADER.encode(), self._session_id.encode())]
        )

    # --- Lifecycle ---------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying channel. Idempotent."""
        if self._flight is not None:
            self._flight.close()
            self._flight = None
        self._channel.close()

    def __enter__(self) -> "RemoteDatabase":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def open_remote(
    endpoint: str,
    *,
    tls: bool,
    credentials: Optional[ChannelCredentials],
) -> RemoteDatabase:
    """Open a :class:`RemoteDatabase` against a `host[:port]` authority.

    `tls` selects a secure (`https`/`grpcs`) versus plaintext (`http`/`grpc`)
    channel; `credentials` decides what identity rides that channel — ``None``
    yields an anonymous channel, a :class:`BearerCredentials` attaches the
    bearer to every call. The secure-vs-plaintext channel construction lives in
    the credential, not here. Mints a fresh per-connection session id, mirroring
    the Rust `RemoteSession::connect` (each connection tenant-isolated).

    The channel-level bearer covers the typed gRPC verbs. The Flight SQL lane
    (:meth:`RemoteDatabase.sql`) is a separate `pyarrow.flight` transport that
    does not yet carry the channel-level bearer — see
    https://github.com/f-inverse/jammi-ai/issues/96.
    """
    session_id = str(uuid.uuid4())
    channel = (credentials or AnonymousCredentials()).open_channel(endpoint, tls=tls)
    return RemoteDatabase(channel, session_id=session_id, endpoint=endpoint, tls=tls)


# `google.protobuf.empty_pb2.Empty` is what the no-argument session RPCs take;
# import lazily-cached at module load so each call reuses one instance.
def _empty():
    from google.protobuf import empty_pb2

    return empty_pb2.Empty()


# The Flight SQL type URL for a plain "run this SQL string" command. The server
# (DataFusion's Flight SQL service) dispatches on this `Any` type URL.
_STATEMENT_QUERY_TYPE_URL = (
    b"type.googleapis.com/arrow.flight.protocol.sql.CommandStatementQuery"
)


def _proto_len_delimited(field_number: int, payload: bytes) -> bytes:
    """Encode one length-delimited (wire type 2) protobuf field.

    `tag = (field_number << 3) | 2`, then the varint length, then the bytes.
    Only the two messages this client constructs — `Any` and
    `CommandStatementQuery` — are encoded here, both of which use exactly this
    one wire type, so a full protobuf runtime is unnecessary.
    """
    tag = (field_number << 3) | 2
    return bytes([tag]) + _varint(len(payload)) + payload


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _command_statement_query(query: str) -> bytes:
    """Build the `google.protobuf.Any`-wrapped `CommandStatementQuery` bytes a
    Flight SQL `get_flight_info` command carries for a SQL string.

    `CommandStatementQuery { query = 1 }` → wrapped in
    `Any { type_url = 1, value = 2 }`.
    """
    inner = _proto_len_delimited(1, query.encode("utf-8"))
    any_msg = _proto_len_delimited(1, _STATEMENT_QUERY_TYPE_URL) + _proto_len_delimited(
        2, inner
    )
    return any_msg
