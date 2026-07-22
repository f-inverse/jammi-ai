from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PerQueryAudit(_message.Message):
    __slots__ = ("query_id", "tenant_id", "model_id", "model_version", "query_lineage", "top_k_result_ids", "retrieval_scores", "executed_at_micros", "signature")
    QUERY_ID_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    QUERY_LINEAGE_FIELD_NUMBER: _ClassVar[int]
    TOP_K_RESULT_IDS_FIELD_NUMBER: _ClassVar[int]
    RETRIEVAL_SCORES_FIELD_NUMBER: _ClassVar[int]
    EXECUTED_AT_MICROS_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    query_id: str
    tenant_id: str
    model_id: str
    model_version: str
    query_lineage: str
    top_k_result_ids: _containers.RepeatedScalarFieldContainer[str]
    retrieval_scores: _containers.RepeatedScalarFieldContainer[float]
    executed_at_micros: int
    signature: str
    def __init__(self, query_id: _Optional[str] = ..., tenant_id: _Optional[str] = ..., model_id: _Optional[str] = ..., model_version: _Optional[str] = ..., query_lineage: _Optional[str] = ..., top_k_result_ids: _Optional[_Iterable[str]] = ..., retrieval_scores: _Optional[_Iterable[float]] = ..., executed_at_micros: _Optional[int] = ..., signature: _Optional[str] = ...) -> None: ...

class AuditLogRequest(_message.Message):
    __slots__ = ("records",)
    RECORDS_FIELD_NUMBER: _ClassVar[int]
    records: _containers.RepeatedCompositeFieldContainer[PerQueryAudit]
    def __init__(self, records: _Optional[_Iterable[_Union[PerQueryAudit, _Mapping]]] = ...) -> None: ...

class AuditFetchByQueryIdRequest(_message.Message):
    __slots__ = ("query_id",)
    QUERY_ID_FIELD_NUMBER: _ClassVar[int]
    query_id: str
    def __init__(self, query_id: _Optional[str] = ...) -> None: ...

class AuditFetchByQueryIdResponse(_message.Message):
    __slots__ = ("record",)
    RECORD_FIELD_NUMBER: _ClassVar[int]
    record: PerQueryAudit
    def __init__(self, record: _Optional[_Union[PerQueryAudit, _Mapping]] = ...) -> None: ...

class AuditFetchRecentRequest(_message.Message):
    __slots__ = ("limit",)
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    limit: int
    def __init__(self, limit: _Optional[int] = ...) -> None: ...

class AuditFetchRecentResponse(_message.Message):
    __slots__ = ("records",)
    RECORDS_FIELD_NUMBER: _ClassVar[int]
    records: _containers.RepeatedCompositeFieldContainer[PerQueryAudit]
    def __init__(self, records: _Optional[_Iterable[_Union[PerQueryAudit, _Mapping]]] = ...) -> None: ...
