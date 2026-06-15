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

import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import grpc
import pyarrow as pa
import pyarrow.flight  # noqa: F401  (registers the `pa.flight` submodule)

from . import _conformal
from ._assembly import (
    _CALIBRATION_SHAPE,
    _CHANNEL_COLUMN_TYPE,
    _EDGE_DIRECTION,
    _EVAL_TASK,
    _FILE_FORMAT,
    _MODEL_TASK,
    _PROPAGATION_OUTPUT,
    _PROPAGATION_WEIGHTING,
    _SET_AGGREGATOR,
    _SOURCE_KIND_NAME,
    _channel_columns_message,
    _local_source_url,
    _modality_value,
    build_context_predictor_request,
    build_fine_tune_graph_request,
    build_fine_tune_request,
)
from ._credentials import AnonymousCredentials, ChannelCredentials
from ._errors import TrainingError
from ._generated.jammi.v1 import catalog_pb2, catalog_pb2_grpc
from ._generated.jammi.v1 import embedding_pb2, embedding_pb2_grpc
from ._generated.jammi.v1 import eval_pb2, eval_pb2_grpc
from ._generated.jammi.v1 import inference_pb2, inference_pb2_grpc
from ._generated.jammi.v1 import pipeline_pb2, pipeline_pb2_grpc
from ._generated.jammi.v1 import training_pb2, training_pb2_grpc
from ._generated.jammi.v1 import trigger_pb2, trigger_pb2_grpc

# The header carrying a connection's opaque session id. The server's tenant
# interceptor reads it on every request and resolves the bound tenant; it is the
# same key the Rust SDK and npm client use. Kept in one place so the value never
# drifts across the seam.
SESSION_HEADER = "jammi-session-id"


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


def _encode_ipc_schema(schema: pa.Schema) -> bytes:
    """Encode a `pyarrow.Schema` as a schema-only Arrow IPC stream — the framing
    the engine's `decode_ipc_schema` reads back (a `StreamWriter` opened on the
    schema and finished with no batches). This is the byte shape
    `MutableTableDefinition.schema` and `RegisterTopicRequest.schema` carry, the
    same self-describing IPC framing the trigger batch payloads use.
    """
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema):
        # No batches: a schema-only stream is the writer opened and closed, which
        # emits the schema message the server decodes back into a `SchemaRef`.
        pass
    return sink.getvalue().to_pybytes()


def _table_to_arrow_batch(batch: pa.Table) -> "trigger_pb2.ArrowBatch":
    """Encode a `pyarrow.Table` into one `ArrowBatch` carrying a self-describing
    Arrow IPC stream in `data_body` (schema header inline, `data_header` empty) —
    the Flight-IPC framing the engine's `decode_ipc_stream` reads back, and the
    inverse of :func:`_arrow_batch_to_table`. The whole table rides as one logical
    event: a multi-chunk table is collapsed to a single chunk first (mirroring the
    embedded `publish_topic`, which `concat_batches` the streamed chunks), so the
    stream carries exactly one RecordBatch message — the count the engine's
    `decode_publish_batch` requires.
    """
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_table(batch.combine_chunks())
    return trigger_pb2.ArrowBatch(data_header=b"", data_body=sink.getvalue().to_pybytes())


def _mutable_table_definition_to_dict(
    d: catalog_pb2.MutableTableDefinition,
) -> Dict[str, Any]:
    """Project a wire `MutableTableDefinition` into the embed wheel's dict shape.

    Mirrors the embedded `Database.list_mutable_tables` entry — `id`, `schema`
    (a `pyarrow.Schema` decoded from the IPC framing), `primary_key`, `indexes`
    (each a `{"name", "columns", "unique"}` dict), `order_column` (empty string
    when the table declares none, matching the embedded `unwrap_or_default`), and
    `chunk_size` — so a caller reads the same keys regardless of transport.
    """
    schema = pa.ipc.open_stream(d.schema).schema
    return {
        "id": d.id,
        "schema": schema,
        "primary_key": list(d.primary_key),
        "indexes": [
            {"name": idx.name, "columns": list(idx.columns), "unique": idx.unique}
            for idx in d.indexes
        ],
        "order_column": d.order_column,
        "chunk_size": d.chunk_size,
    }


def _aggregate_metrics_to_dict(m: eval_pb2.AggregateMetrics) -> Dict[str, Any]:
    """Project wire `AggregateMetrics` into the embed wheel's aggregate dict."""
    return {
        "recall_at_k": m.recall_at_k,
        "precision_at_k": m.precision_at_k,
        "mrr": m.mrr,
        "ndcg": m.ndcg,
    }


def _per_query_record_to_dict(r: eval_pb2.PerQueryRecord) -> Dict[str, Any]:
    """Project a wire `PerQueryRecord` into the embed wheel's per-query dict.

    `recall_at_ks` is a list of two-element `[k, recall]` pairs — the engine's
    `(usize, f64)` tuples serialize to JSON arrays, so the typed `RecallAtK`
    messages are flattened back to that pair shape. `cohorts` is a plain dict
    (`{}` when the query carried no tags).
    """
    return {
        "query_id": r.query_id,
        "metrics": {
            "recall": r.metrics.recall,
            "precision": r.metrics.precision,
            "mrr": r.metrics.mrr,
            "ndcg": r.metrics.ndcg,
        },
        "recall_at_ks": [[p.k, p.recall] for p in r.recall_at_ks],
        "distance": r.distance,
        "cohorts": dict(r.cohorts),
    }


def _embedding_report_to_dict(report: eval_pb2.EmbeddingEvalReport) -> Dict[str, Any]:
    """Project a wire `EmbeddingEvalReport` into the embed wheel's nested dict.

    The shape matches the embed `Database.eval_embeddings` (which serializes the
    engine's `EmbeddingEvalReport` to a dict): ``eval_run_id`` plus an
    ``aggregate`` block and the ``per_query`` records — so a caller reads the
    same keys on both transports.
    """
    return {
        "eval_run_id": report.eval_run_id,
        "aggregate": _aggregate_metrics_to_dict(report.aggregate),
        "per_query": [_per_query_record_to_dict(r) for r in report.per_query],
    }


def _inference_aggregate_to_dict(agg: eval_pb2.InferenceAggregate) -> Dict[str, Any]:
    """Project a wire `InferenceAggregate` into the embed wheel's tagged dict.

    The engine's enum is internally serde-tagged (`"task": "classification"` /
    `"task": "ner"`), so the proto oneof is flattened into one dict carrying the
    tag alongside the variant's fields — not nested under a variant key.
    """
    which = agg.WhichOneof("aggregate")
    if which == "classification":
        c = agg.classification
        return {
            "task": "classification",
            "accuracy": c.accuracy,
            "f1": c.f1,
            "per_class": {
                label: {"precision": m.precision, "recall": m.recall, "f1": m.f1}
                for label, m in c.per_class.items()
            },
        }
    if which == "ner":
        n = agg.ner
        return {
            "task": "ner",
            "precision": n.precision,
            "recall": n.recall,
            "f1": n.f1,
            "per_type": {
                t: {
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "support": m.support,
                }
                for t, m in n.per_type.items()
            },
        }
    raise RuntimeError("InferenceEvalReport carried no aggregate")


def _entity_to_dict(e: eval_pb2.Entity) -> Dict[str, Any]:
    """Project a wire `Entity` span into the embed wheel's entity dict."""
    return {
        "label": e.label,
        "start": e.start,
        "end": e.end,
        "text": e.text,
        "confidence": e.confidence,
    }


def _per_record_prediction_to_dict(
    rec: eval_pb2.PerRecordPrediction,
) -> Dict[str, Any]:
    """Project a wire `PerRecordPrediction` into the embed wheel's tagged dict.

    Same flattening as :func:`_inference_aggregate_to_dict`: the oneof becomes
    one dict with a ``"task"`` tag alongside the variant's fields.
    """
    which = rec.WhichOneof("prediction")
    if which == "classification":
        c = rec.classification
        return {
            "task": "classification",
            "record_id": c.record_id,
            "predicted": c.predicted,
            "gold": c.gold,
        }
    if which == "ner":
        n = rec.ner
        return {
            "task": "ner",
            "record_id": n.record_id,
            "predicted": [_entity_to_dict(e) for e in n.predicted],
            "gold": [_entity_to_dict(e) for e in n.gold],
        }
    raise RuntimeError("PerRecordPrediction carried no prediction")


def _inference_report_to_dict(report: eval_pb2.InferenceEvalReport) -> Dict[str, Any]:
    """Project a wire `InferenceEvalReport` into the embed wheel's nested dict."""
    return {
        "aggregate": _inference_aggregate_to_dict(report.aggregate),
        "per_record": [_per_record_prediction_to_dict(r) for r in report.per_record],
    }


def _metric_significance_to_dict(s: eval_pb2.MetricSignificance) -> Dict[str, Any]:
    """Project a wire `MetricSignificance` into the embed wheel's dict."""
    return {"p_value": s.p_value, "ci_lower": s.ci_lower, "ci_upper": s.ci_upper}


def _aggregate_delta_to_dict(d: eval_pb2.AggregateDelta) -> Dict[str, Any]:
    """Project a wire `AggregateDelta` into the embed wheel's delta dict.

    ``significance`` is presence-wrapped on the wire: when the baseline and
    treatment runs share no query to pair on, the embed dict carries ``None``
    there — the key is always present, mirroring the engine's serde output.
    """

    def metric(m: eval_pb2.MetricDelta) -> Dict[str, Any]:
        return {"absolute": m.absolute, "relative": m.relative}

    significance = None
    if d.HasField("significance"):
        s = d.significance
        significance = {
            "recall_at_k": _metric_significance_to_dict(s.recall_at_k),
            "precision_at_k": _metric_significance_to_dict(s.precision_at_k),
            "mrr": _metric_significance_to_dict(s.mrr),
            "ndcg": _metric_significance_to_dict(s.ndcg),
        }
    return {
        "recall_at_k": metric(d.recall_at_k),
        "precision_at_k": metric(d.precision_at_k),
        "mrr": metric(d.mrr),
        "ndcg": metric(d.ndcg),
        "significance": significance,
    }


def _compare_report_to_dict(report: eval_pb2.CompareEvalReport) -> Dict[str, Any]:
    """Project a wire `CompareEvalReport` into the embed wheel's nested dict.

    ``delta`` is presence-wrapped on the wire: the baseline (first) table carries
    ``None`` — the key is always present, mirroring the engine's serde output.
    """
    return {
        "per_table": [
            {
                "table_name": t.table_name,
                "embedding_eval": _embedding_report_to_dict(t.embedding_eval),
                "delta": (
                    _aggregate_delta_to_dict(t.delta) if t.HasField("delta") else None
                ),
            }
            for t in report.per_table
        ]
    }


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


# Inverse of `_CHANNEL_COLUMN_TYPE`: wire enum → canonical PascalCase token. The
# inverse of one map keeps the two directions from drifting; `list_channels`
# decodes a wire column's `data_type` back to the same token `register_channel`
# accepts, matching the embed binding's `ChannelColumnType::as_str`.
_CHANNEL_COLUMN_TYPE_NAME = {v: k for k, v in _CHANNEL_COLUMN_TYPE.items()}

# Terminal training-job states, matching the engine's `TrainingJobStatus`.
_TERMINAL_STATES = {"completed", "failed"}


def _channel_spec_to_dict(c: catalog_pb2.Channel) -> Dict[str, Any]:
    """Project a wire `Channel` into the embed binding's channel-spec dict.

    The shape matches the embed `Database.list_channels` entry — ``channel_id``,
    ``priority``, and ``columns`` (each a ``{"name", "data_type"}`` dict whose
    ``data_type`` is the canonical PascalCase token) — so a caller reads the same
    keys, in the same column order, regardless of transport.
    """
    return {
        "channel_id": c.channel_id,
        "priority": c.priority,
        "columns": [
            {"name": col.name, "data_type": _CHANNEL_COLUMN_TYPE_NAME[col.data_type]}
            for col in c.columns
        ],
    }


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
        # The trigger data-plane (`publish_topic` / `subscribe_collect`). The
        # topic-admin lifecycle (register/drop/list) is control-plane and rides
        # `self._catalog`; only the publish/subscribe compute verbs hit this stub.
        self._trigger = trigger_pb2_grpc.TriggerServiceStub(channel)
        # The Flight SQL client shares the gRPC endpoint (the server co-mounts
        # Flight SQL and the typed gRPC services on one Tonic port); built lazily
        # on first `sql` so a connection that never queries SQL pays nothing.
        self._flight = None

    @property
    def session_id(self) -> str:
        """The opaque session id the server keys this connection's tenant by."""
        return self._session_id

    # --- Catalog: tenant trio + handshake ---------------------------------------

    def set_tenant(self, tenant_id: str) -> None:
        """Set the sticky tenant scope on this connection. Pass an empty string
        to clear.

        The scope stays in effect server-side until the next `set_tenant`
        replaces it. This is a setter — it returns ``None``, not a fresh scoped
        handle. For a block-scoped binding that restores the prior tenant on
        exit, use :meth:`tenant_scope`.

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

    @contextmanager
    def tenant_scope(self, tenant_id: str) -> Iterator["RemoteDatabase"]:
        """Scope this connection to ``tenant_id`` for the duration of a ``with``
        block, restoring the prior tenant on exit.

        ::

            with db.tenant_scope("a"):
                db.list_sources()   # sees tenant-a + global rows only
                db.sql("...")       # discriminator-column rows scoped to a
            # prior scope restored here

        Nesting restores correctly — after an inner ``with db.tenant_scope("b")``
        block exits, the outer ``"a"`` scope is back in effect. The remote tenant
        lives server-side (keyed by this session id); the client captures the
        prior tenant locally via `CatalogService.GetTenant` on entry and rebinds
        it (or clears, if the prior scope was unscoped) on exit, rather than
        blindly clearing to unscoped.
        """
        prior = self.tenant()
        self.set_tenant(tenant_id)
        try:
            yield self
        finally:
            self.set_tenant(prior or "")

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

    # --- Mutable companion tables (control plane) --------------------------------
    #
    # The mutable-table lifecycle is control-plane: register / drop / introspect a
    # mutable companion table against the remote catalog (`CatalogService`). The
    # schema rides as a schema-only Arrow IPC stream (the framing the server's
    # `decode_ipc_schema` reads back); the wire body is tenant-free — the server
    # stamps the session's tenant from the `jammi-session-id` header. Data DML
    # (INSERT / UPDATE / DELETE / SELECT over `mutable.public.<name>`) flows over
    # the Flight SQL lane (:meth:`sql`), not these verbs. The signatures mirror the
    # embed `Database`'s so a caller swaps transports without changing the call.

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
        """Register a mutable companion table on the remote engine and return its
        catalog id.

        `schema` is the table's `pyarrow.Schema`; `primary_key` is a non-empty
        list of column names drawn from it. `indexes` is an optional list of dicts
        of shape ``{"name": str, "columns": [str, ...], "unique": bool=False}`` —
        one secondary index per entry. `order_column` is an optional monotonic
        column enabling streaming `scan_after` reads; `chunk_size` overrides the
        engine's default scan chunk size. Tenant scope is the session's bound
        tenant (the wire body is tenant-free). Maps to
        `CatalogService.CreateMutableTable`.
        """
        wire_indexes = [
            catalog_pb2.MutableIndex(
                name=idx["name"],
                columns=list(idx["columns"]),
                unique=bool(idx.get("unique", False)),
            )
            for idx in (indexes or [])
        ]
        definition = catalog_pb2.MutableTableDefinition(
            id=name,
            schema=_encode_ipc_schema(schema),
            primary_key=list(primary_key),
            indexes=wire_indexes,
        )
        if order_column is not None:
            definition.order_column = order_column
        if chunk_size is not None:
            definition.chunk_size = chunk_size
        resp = self._catalog.CreateMutableTable(
            catalog_pb2.CreateMutableTableRequest(definition=definition),
            metadata=self._metadata,
        )
        return resp.mutable_table_id

    def drop_mutable_table(self, name: str, *, if_exists: bool = False) -> None:
        """Drop a mutable companion table by id. With ``if_exists=True``, dropping
        a table that is not registered is a no-op; otherwise the engine's typed
        NotFound is surfaced. Maps to `CatalogService.DropMutableTable`.
        """
        try:
            self._catalog.DropMutableTable(
                catalog_pb2.DropMutableTableRequest(mutable_table_id=name),
                metadata=self._metadata,
            )
        except grpc.RpcError as exc:
            if if_exists and exc.code() == grpc.StatusCode.NOT_FOUND:
                return
            raise

    def list_mutable_tables(self) -> List[Dict[str, Any]]:
        """A descriptor for every mutable companion table registered to the current
        tenant. Each entry is the same dict shape as the embed
        `Database.list_mutable_tables`: ``id``, ``schema`` (a `pyarrow.Schema`),
        ``primary_key``, ``indexes``, ``order_column`` (empty when none), and
        ``chunk_size``. Maps to `CatalogService.ListMutableTables`.
        """
        resp = self._catalog.ListMutableTables(
            catalog_pb2.ListMutableTablesRequest(), metadata=self._metadata
        )
        return [_mutable_table_definition_to_dict(d) for d in resp.definitions]

    # --- Topics: admin (control plane) ------------------------------------------
    #
    # The topic-admin lifecycle is control-plane (`CatalogService`): register /
    # drop / list a trigger-stream topic. The publish/subscribe compute verbs are
    # data-plane and ride `TriggerService` (below). The drop/list wire is id-keyed
    # (the server resolves the topic by `TopicId`), while the embed surface is
    # name-keyed — so :meth:`drop_topic` resolves the name → id via `ListTopics`
    # first, matching the embedded tenant-scoped `lookup_by_name`.

    def register_topic(
        self,
        name: str,
        *,
        schema: pa.Schema,
        broker_metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a trigger-stream topic on the remote engine and return its
        engine-minted topic id.

        `schema` is the contract every published batch must satisfy (rides as a
        schema-only Arrow IPC stream). `broker_metadata` is opaque driver-side
        configuration (retention, replication, …). Tenant scope is the session's
        bound tenant. Maps to `CatalogService.RegisterTopic`.
        """
        request = catalog_pb2.RegisterTopicRequest(
            name=name,
            schema=_encode_ipc_schema(schema),
            broker_metadata=broker_metadata or {},
        )
        resp = self._catalog.RegisterTopic(request, metadata=self._metadata)
        return resp.topic_id

    def drop_topic(self, name: str, *, if_exists: bool = False) -> None:
        """Drop a trigger-stream topic by name. With ``if_exists=True``, dropping a
        topic that is not registered is a no-op; otherwise a missing topic raises.
        Maps to `CatalogService.DropTopic`.

        The wire drop is id-keyed, but the embed surface (and this verb) is
        name-keyed — so the name is resolved to its id via :meth:`list_topics`
        first, the same tenant-scoped lookup the embedded `drop_topic` does. A
        name that resolves to no topic is the no-op (``if_exists``) or a
        ``ValueError`` (matching the embedded NotFound), without a wire hop.
        """
        topic_id = self._resolve_topic_id(name)
        if topic_id is None:
            if if_exists:
                return
            raise ValueError(f"topic '{name}' not found")
        try:
            self._catalog.DropTopic(
                catalog_pb2.DropTopicRequest(topic_id=topic_id, if_exists=if_exists),
                metadata=self._metadata,
            )
        except grpc.RpcError as exc:
            if if_exists and exc.code() == grpc.StatusCode.NOT_FOUND:
                return
            raise

    def list_topics(self) -> List[str]:
        """The name of every topic visible to the current tenant binding. Maps to
        `CatalogService.ListTopics`; same list-of-names shape the embed
        `Database.list_topics` returns.
        """
        return [t.name for t in self._list_topics_raw()]

    def _list_topics_raw(self) -> List[catalog_pb2.Topic]:
        """The full `Topic` messages for the current tenant — the shared read
        backing both :meth:`list_topics` (names) and the name→id resolution
        :meth:`drop_topic` needs."""
        resp = self._catalog.ListTopics(
            catalog_pb2.ListTopicsRequest(), metadata=self._metadata
        )
        return list(resp.topics)

    def _resolve_topic_id(self, name: str) -> Optional[str]:
        """Resolve a topic name to its id within the current tenant scope, or
        ``None`` if no such topic exists — the client-side analogue of the engine's
        tenant-scoped `lookup_by_name`."""
        for topic in self._list_topics_raw():
            if topic.name == name:
                return topic.topic_id
        return None

    # --- Topics: pub/sub (data plane) -------------------------------------------
    #
    # The publish/subscribe compute verbs ride `TriggerService` (the data plane).
    # Both resolve nothing client-side beyond the topic name the server looks up;
    # the batch payloads ride as `ArrowBatch` (Flight IPC framing). The signatures
    # mirror the embed `Database`'s so a caller swaps transports unchanged.

    def publish_topic(self, topic: str, *, batch: pa.Table) -> int:
        """Publish one batch of rows to a topic and return the engine-assigned
        offset.

        `batch` is a `pyarrow.Table` whose schema must match the topic's; it rides
        as one `ArrowBatch` (a multi-chunk table publishes as one logical event,
        concatenated server-side — matching the embedded `publish_topic`). The
        publish is scoped to the session's bound tenant. Maps to
        `TriggerService.Publish`.
        """
        request = trigger_pb2.PublishRequest(
            topic=trigger_pb2.TopicName(name=topic),
            batch=_table_to_arrow_batch(batch),
        )
        resp = self._trigger.Publish(request, metadata=self._metadata)
        return resp.offset

    def subscribe_collect(
        self,
        topic: str,
        *,
        predicate: Optional[str] = None,
        from_offset: Optional[int] = None,
        max_batches: int = 64,
    ) -> pa.Table:
        """Open a subscription, collect up to `max_batches` matching batches
        (replay + live tail joined), then close — returning the concatenated
        payload as a `pyarrow.Table`.

        This mirrors the embedded `Database.subscribe_collect` exactly: it drives
        the open-ended `Subscribe` stream (``replay_only`` unset — replay AND live
        tail, NOT a bounded replay-only drain), accumulates up to `max_batches`
        delivered batches, then cancels the gRPC call client-side so the server's
        tail task stops (its send fails on the cancel and the spawned forwarder
        breaks) rather than leaking. `predicate` is an optional SQL filter applied
        server-side; `from_offset` starts the replay at an offset (unset == live
        tail only). Maps to `TriggerService.Subscribe`.
        """
        request = trigger_pb2.SubscribeRequest(
            topic=trigger_pb2.TopicName(name=topic),
            predicate=predicate or "",
        )
        if from_offset is not None:
            request.from_offset = from_offset

        call = self._trigger.Subscribe(request, metadata=self._metadata)
        collected: List[pa.Table] = []
        try:
            # Mirror the embedded `while out.len() < max_batches { next() }`: pull a
            # batch only while under the bound, so a satisfied collect never blocks
            # on one more read of the live tail. The stream ends on its own only if
            # the server closes it (it does not for a live-tail subscription), so
            # `max_batches` is the terminator — identical to the embedded contract.
            stream = iter(call)
            while len(collected) < max_batches:
                try:
                    delivered = next(stream)
                except StopIteration:
                    break
                collected.append(_arrow_batch_to_table(delivered.batch))
        finally:
            # Cancel client-side so the server's tail task observes the closed
            # stream and stops — the bounded collect leaves no live subscription
            # leaking server-side, the same close the embedded `subscribe_collect`
            # gets by dropping the engine stream.
            call.cancel()

        if not collected:
            return pa.table({})
        return pa.concat_tables(collected)

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
        embedding_table: Optional[str] = None,
    ) -> pa.Table:
        """Nearest-neighbor search over a source's embedding table.

        `query` is the query vector; `filter` is an optional SQL predicate over
        the hydrated results; `select` projects columns (empty keeps the
        keyed+scored shape). `embedding_table` names which of the source's
        embedding tables to search (e.g. a raw, propagated, or fine-tuned
        table); ``None`` searches the most-recent ready table. Returns a
        `pyarrow.Table`. Maps to `EmbeddingService.Search`.
        """
        request = embedding_pb2.SearchRequest(
            source_id=source,
            query_vector=embedding_pb2.QueryVector(values=list(query)),
            k=k,
            select=list(select or []),
        )
        if filter is not None:
            request.filter = filter
        if embedding_table is not None:
            request.embedding_table = embedding_table
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
        seed: Optional[int] = None,
        regression_loss: Optional[str] = None,
        regression_beta: Optional[float] = None,
        quantile_levels: Optional[List[float]] = None,
    ) -> RemoteTrainingJob:
        """Submit a LoRA fine-tuning job to the remote engine; poll the handle.

        Returns a :class:`RemoteTrainingJob` — same handle shape and verb
        signature as the embed `Database.fine_tune`. Maps to
        `TrainingService.StartTraining` with the `FineTuneSpec` arm; all config
        kwargs are optional, applying the engine defaults when omitted.
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
        seed: Optional[int] = None,
    ) -> RemoteTrainingJob:
        """Submit a graph-supervised fine-tune (S11) to the remote engine.

        Returns a :class:`RemoteTrainingJob`, mirroring the embed
        `Database.fine_tune_graph`. Maps to `TrainingService.StartTraining` with
        the `GraphFineTuneSpec` arm. `edge_provenance` is the load-bearing
        circularity distinction — "declared" external edges teach the metric
        something new; "similarity" edges are a weak bootstrap only.
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

    # --- Inference (data plane) ---------------------------------------------------
    #
    # The bulk inference verb DOES hit the wire: the model and the registered
    # source both live in the engine, so the compute runs where the data is
    # (`InferenceService.Infer`) and only the output rows cross the wire. The
    # signature mirrors the embed `Database`'s so a caller swaps transports
    # without changing the call.

    def infer(
        self,
        *,
        source: str,
        model: str,
        columns: List[str],
        task: str,
        key: str,
    ) -> pa.Table:
        """Run `model` over `columns` of a registered source for `task`,
        returning one output row per source row as a `pyarrow.Table`.

        `task` is the snake-case model-task string (``"text_embedding"``,
        ``"image_embedding"``, ``"audio_embedding"``, ``"classification"``,
        ``"ner"``, ``"regression"``); `key` names the column whose value becomes
        each output row's ``_row_id``. Maps to `InferenceService.Infer`. The
        whole result rides back as one unary `ArrowBatch` (a single Arrow IPC
        stream), so gRPC's default 4 MB per-message receive cap bounds the
        result size a default channel can carry.
        """
        resp = self._inference.Infer(
            inference_pb2.InferRequest(
                source_id=source,
                model_id=model,
                task=_MODEL_TASK[task],
                columns=list(columns),
                key_column=key,
            ),
            metadata=self._metadata,
        )
        return _arrow_batch_to_table(resp.result)

    # --- Engine-state pipeline verbs (PipelineService) ----------------------------
    #
    # These build durable graph/embedding artifacts or assemble a target's
    # conditioning context on the remote engine. The two graph-build verbs return
    # the materialised table name (the caller reads it via `sql(...)`); the
    # compute stays server-side, so the table is byte-identical to one built in
    # the embedded engine. `assemble_context` returns the pooled vector inline.
    # The signatures mirror the embed `Database`'s so a caller swaps transports
    # without changing the call.

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

    # --- Evaluation verbs (EvalService) -------------------------------------------
    #
    # The eval family measures model quality against golden data held in the
    # engine, so the compute runs where the data is and only the typed report
    # crosses the wire. Each verb returns the same nested dict the embed
    # `Database` produces (the engine's serde report shape), so a caller reads
    # the same keys on both transports. A `golden_source` is an unquoted
    # relation reference — a bare name or the full catalog path
    # `<source>.public.<table>`.

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

        Returns the same nested dict the embed `Database` produces:
        ``eval_run_id``, ``aggregate`` (mean ``recall_at_k`` /
        ``precision_at_k`` / ``mrr`` / ``ndcg`` over all queries), and
        ``per_query`` (one record per golden-set query). ``embedding_table``
        names the result table to evaluate; ``None`` resolves the source's most
        recent embedding table. ``golden_source`` addresses the golden set by
        its full catalog path (``<source>.public.<table>``) or bare name.
        ``cohorts`` optionally maps a golden-set ``query_id`` to an opaque
        ``{key: value}`` segment map, persisted with that query's per-query
        metrics (read back via :meth:`eval_per_query`). Maps to
        `EvalService.EvalEmbeddings`.
        """
        request = eval_pb2.EvalEmbeddingsRequest(
            source_id=source,
            golden_source=golden_source,
            k=k,
        )
        if embedding_table is not None:
            request.embedding_table = embedding_table
        for query_id, tags in (cohorts or {}).items():
            request.cohorts[query_id].tags.update(tags)
        resp = self._eval.EvalEmbeddings(request, metadata=self._metadata)
        return _embedding_report_to_dict(resp)

    def eval_per_query(self, eval_run_id: str) -> List[Dict[str, Any]]:
        """Read back the persisted per-query eval records for a run, scoped to
        the calling tenant.

        Returns a list of dicts, each carrying ``eval_run_id``, ``query_id``,
        ``cohorts`` (a dict), and ``metrics`` (a dict of ``recall@1/3/5/10``,
        ``mrr``, ``ndcg``, ``distance``) — the same shape the embed `Database`
        produces. The wire carries ``cohorts``/``metrics`` as JSON-object
        strings (their storage shape); they are decoded into structured dicts
        here, exactly as the embed binding does. Maps to
        `EvalService.EvalPerQuery`.
        """
        resp = self._eval.EvalPerQuery(
            eval_pb2.EvalPerQueryRequest(eval_run_id=eval_run_id),
            metadata=self._metadata,
        )
        return [
            {
                "eval_run_id": rec.eval_run_id,
                "query_id": rec.query_id,
                "cohorts": json.loads(rec.cohorts_json),
                "metrics": json.loads(rec.metrics_json),
            }
            for rec in resp.records
        ]

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

        Returns the same nested dict the embed `Database` produces:
        ``aggregate`` (task-shaped, tagged by ``"task"``) and ``per_record``
        (one tagged record per predicted/gold pair). ``task`` is
        ``"classification"`` or ``"ner"``; ``golden_source`` addresses the
        golden labels by full catalog path (``<source>.public.<table>``) or
        bare name, with ``label_column`` naming the gold-label column. Maps to
        `EvalService.EvalInference`.
        """
        try:
            task_value = _EVAL_TASK[task]
        except KeyError:
            raise ValueError(
                f"task must be 'classification' or 'ner' (got {task!r})"
            ) from None
        resp = self._eval.EvalInference(
            eval_pb2.EvalInferenceRequest(
                model_id=model,
                source_id=source,
                columns=list(columns),
                task=task_value,
                golden_source=golden_source,
                label_column=label_column,
            ),
            metadata=self._metadata,
        )
        return _inference_report_to_dict(resp)

    def eval_compare(
        self,
        *,
        embedding_tables: List[str],
        source: str,
        golden_source: str,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Compare multiple embedding tables side-by-side against one golden set.

        Returns the same nested dict the embed `Database` produces: a
        ``per_table`` list whose first entry is the baseline (``delta: None``)
        and whose subsequent entries carry a ``delta`` against it (per-metric
        absolute/relative deltas plus paired ``significance``, ``None`` when
        the runs share no query to pair on). ``golden_source`` addresses the
        golden set by full catalog path (``<source>.public.<table>``) or bare
        name. Maps to `EvalService.EvalCompare`.
        """
        resp = self._eval.EvalCompare(
            eval_pb2.EvalCompareRequest(
                embedding_tables=list(embedding_tables),
                source_id=source,
                golden_source=golden_source,
                k=k,
            ),
            metadata=self._metadata,
        )
        return _compare_report_to_dict(resp)

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

    # --- Channels ----------------------------------------------------------------

    def register_channel(
        self,
        channel_id: str,
        *,
        priority: int,
        columns: Sequence[Tuple[str, str]],
    ) -> None:
        """Register an evidence-provenance channel and its initial columns.

        `columns` is a list of `(name, dtype)` tuples where `dtype` is one of
        ``"Float32"``, ``"Float64"``, ``"Int32"``, ``"Int64"``, ``"Utf8"``,
        ``"Boolean"`` — the same vocabulary the embed binding accepts.
        ``priority`` orders the channel against others contributing the same
        column. The channel id is unique PER TENANT, scoped to this connection's
        currently bound tenant: two tenants may each register a channel of the
        same id without collision, but re-registering an id already present for
        the bound tenant raises. Maps to `CatalogService.RegisterChannel`.
        """
        self._catalog.RegisterChannel(
            catalog_pb2.RegisterChannelRequest(
                channel_id=channel_id,
                priority=priority,
                columns=_channel_columns_message(columns),
            ),
            metadata=self._metadata,
        )

    def add_channel_columns(
        self,
        channel_id: str,
        *,
        columns: Sequence[Tuple[str, str]],
    ) -> None:
        """Append columns to an already-registered channel (append-only).

        `columns` is a list of `(name, dtype)` tuples in the same vocabulary as
        :meth:`register_channel`. The append-only invariant is enforced
        server-side: redeclaring an existing column with a different dtype
        raises. Maps to `CatalogService.AddChannelColumns`.
        """
        self._catalog.AddChannelColumns(
            catalog_pb2.AddChannelColumnsRequest(
                channel_id=channel_id,
                columns=_channel_columns_message(columns),
            ),
            metadata=self._metadata,
        )

    def list_channels(self) -> List[Dict[str, Any]]:
        """List every evidence channel registered to the currently bound tenant,
        ordered by ``(priority, channel_id)``.

        Returns the same list of dicts the embed `Database` produces — each
        ``{"channel_id", "priority", "columns": [{"name", "data_type"}]}`` with
        ``data_type`` the canonical PascalCase token and ``columns`` in
        declaration order. An unbound connection sees only the global
        (NULL-tenant) channels. Maps to `CatalogService.ListChannels`.
        """
        resp = self._catalog.ListChannels(
            catalog_pb2.ListChannelsRequest(), metadata=self._metadata
        )
        return [_channel_spec_to_dict(c) for c in resp.channels]

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
