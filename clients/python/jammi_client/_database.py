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

import uuid
from typing import Any, Dict, List, Optional, Union

import grpc
import pyarrow as pa

from ._generated.jammi.v1 import embedding_pb2, embedding_pb2_grpc
from ._generated.jammi.v1 import session_pb2, session_pb2_grpc

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
    embedding_pb2.SourceKind.SOURCE_KIND_FILE: "File",
    embedding_pb2.SourceKind.SOURCE_KIND_POSTGRES: "Postgres",
    embedding_pb2.SourceKind.SOURCE_KIND_MYSQL: "MySql",
}

# File-format string → wire `FileFormat` enum. Mirrors the engine's `FileFormat`
# parse so `add_source(format=...)` accepts the same vocabulary as the embed
# wheel's local path.
_FILE_FORMAT = {
    "parquet": embedding_pb2.FileFormat.FILE_FORMAT_PARQUET,
    "csv": embedding_pb2.FileFormat.FILE_FORMAT_CSV,
    "json": embedding_pb2.FileFormat.FILE_FORMAT_JSON,
    "avro": embedding_pb2.FileFormat.FILE_FORMAT_AVRO,
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


def _source_descriptor_to_dict(d: embedding_pb2.SourceDescriptor) -> Dict[str, Any]:
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


class RemoteDatabase:
    """A Database driving a remote jammi engine over the `jammi.v1` gRPC wire.

    The verbs are the transport-agnostic surface; an embedded `Database` (in the
    `jammi-ai` wheel) and a `RemoteDatabase` expose the same vocabulary, only the
    transport differs. Use as a context manager to close the channel on exit.
    """

    def __init__(self, channel: grpc.Channel, *, session_id: str) -> None:
        self._channel = channel
        self._session_id = session_id
        self._metadata = ((SESSION_HEADER, session_id),)
        self._embedding = embedding_pb2_grpc.EmbeddingServiceStub(channel)
        self._session = session_pb2_grpc.SessionServiceStub(channel)

    @property
    def session_id(self) -> str:
        """The opaque session id the server keys this connection's tenant by."""
        return self._session_id

    # --- Session / tenant trio + handshake --------------------------------------

    def with_tenant(self, tenant_id: str) -> None:
        """Bind a tenant scope to this connection. Pass an empty string to clear.

        Maps to `SessionService.SetTenant` / `ClearTenant`, keyed by this
        connection's session id.
        """
        if tenant_id == "":
            self._session.ClearTenant(
                _empty(), metadata=self._metadata
            )
            return
        self._session.SetTenant(
            session_pb2.SetTenantRequest(tenant=session_pb2.Tenant(id=tenant_id)),
            metadata=self._metadata,
        )

    def tenant(self) -> Optional[str]:
        """The tenant currently bound to this connection, or ``None``.

        Maps to `SessionService.GetTenant`.
        """
        resp = self._session.GetTenant(_empty(), metadata=self._metadata)
        return resp.tenant.id or None

    def get_server_info(self) -> Dict[str, Any]:
        """The engine's capabilities handshake: ``version`` / ``features`` /
        ``storage_backends`` / ``services``. Maps to
        `SessionService.GetServerInfo`.

        The first three fields are compile-time facts about the build;
        ``services`` is the runtime tier handshake — the gRPC service tiers this
        deployment mounted (``"core"`` is always present; ``"train"`` /
        ``"event"`` / ``"eval"`` appear only when this server enabled them). A
        client reads ``services`` to know which verbs are reachable here before
        calling them.

        The same keys the embedded `Database.get_server_info` returns, so the
        handshake shape agrees across transports.
        """
        resp = self._session.GetServerInfo(_empty(), metadata=self._metadata)
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
        Maps to `EmbeddingService.AddSource`.
        """
        try:
            file_format = _FILE_FORMAT[format]
        except KeyError:
            raise ValueError(
                f"format must be one of {sorted(_FILE_FORMAT)} (got {format!r})"
            ) from None
        self._embedding.AddSource(
            embedding_pb2.AddSourceRequest(
                source_id=name,
                source_kind=embedding_pb2.SourceKind.SOURCE_KIND_FILE,
                connection=embedding_pb2.SourceConnection(
                    url=_local_source_url(url),
                    format=file_format,
                ),
            ),
            metadata=self._metadata,
        )

    def list_sources(self) -> List[Dict[str, Any]]:
        """A descriptor for every source registered to the current tenant.

        Maps to `EmbeddingService.ListSources`; same dict shape per entry as
        :meth:`describe_source`.
        """
        resp = self._embedding.ListSources(
            embedding_pb2.ListSourcesRequest(), metadata=self._metadata
        )
        return [_source_descriptor_to_dict(d) for d in resp.sources]

    def describe_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Describe one registered source by id, or ``None`` if not visible.

        Maps to `EmbeddingService.DescribeSource`. The engine returns an
        unpopulated descriptor (empty `source_id`) when no such source exists;
        that is surfaced here as ``None``.
        """
        d = self._embedding.DescribeSource(
            embedding_pb2.DescribeSourceRequest(source_id=source_id),
            metadata=self._metadata,
        )
        if not d.source_id:
            return None
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

    # --- Lifecycle ---------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying channel. Idempotent."""
        self._channel.close()

    def __enter__(self) -> "RemoteDatabase":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def open_remote(endpoint: str, *, tls: bool) -> RemoteDatabase:
    """Open a :class:`RemoteDatabase` against a `host[:port]` authority.

    `tls` selects a secure (`https`/`grpcs`) versus plaintext (`http`/`grpc`)
    channel. Mints a fresh per-connection session id, mirroring the Rust
    `RemoteSession::connect` (each connection tenant-isolated).
    """
    session_id = str(uuid.uuid4())
    if tls:
        channel = grpc.secure_channel(endpoint, grpc.ssl_channel_credentials())
    else:
        channel = grpc.insecure_channel(endpoint)
    return RemoteDatabase(channel, session_id=session_id)


# `google.protobuf.empty_pb2.Empty` is what the no-argument session RPCs take;
# import lazily-cached at module load so each call reuses one instance.
def _empty():
    from google.protobuf import empty_pb2

    return empty_pb2.Empty()
