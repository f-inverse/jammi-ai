from . import inference_pb2 as _inference_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Modality(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODALITY_UNSPECIFIED: _ClassVar[Modality]
    TEXT: _ClassVar[Modality]
    IMAGE: _ClassVar[Modality]
    AUDIO: _ClassVar[Modality]
MODALITY_UNSPECIFIED: Modality
TEXT: Modality
IMAGE: Modality
AUDIO: Modality

class GenerateEmbeddingsRequest(_message.Message):
    __slots__ = ("source_id", "model_id", "columns", "key_column", "modality", "cache")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    KEY_COLUMN_FIELD_NUMBER: _ClassVar[int]
    MODALITY_FIELD_NUMBER: _ClassVar[int]
    CACHE_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    model_id: str
    columns: _containers.RepeatedScalarFieldContainer[str]
    key_column: str
    modality: Modality
    cache: _inference_pb2.CachePolicy
    def __init__(self, source_id: _Optional[str] = ..., model_id: _Optional[str] = ..., columns: _Optional[_Iterable[str]] = ..., key_column: _Optional[str] = ..., modality: _Optional[_Union[Modality, str]] = ..., cache: _Optional[_Union[_inference_pb2.CachePolicy, str]] = ...) -> None: ...

class ResultTable(_message.Message):
    __slots__ = ("table_name", "source_id", "model_id", "dimensions", "row_count", "status", "task", "cache_outcome")
    TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    ROW_COUNT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    CACHE_OUTCOME_FIELD_NUMBER: _ClassVar[int]
    table_name: str
    source_id: str
    model_id: str
    dimensions: int
    row_count: int
    status: str
    task: _inference_pb2.ModelTask
    cache_outcome: _inference_pb2.CacheOutcome
    def __init__(self, table_name: _Optional[str] = ..., source_id: _Optional[str] = ..., model_id: _Optional[str] = ..., dimensions: _Optional[int] = ..., row_count: _Optional[int] = ..., status: _Optional[str] = ..., task: _Optional[_Union[_inference_pb2.ModelTask, str]] = ..., cache_outcome: _Optional[_Union[_inference_pb2.CacheOutcome, str]] = ...) -> None: ...

class EncodeQueryRequest(_message.Message):
    __slots__ = ("model_id", "modality", "text", "data")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODALITY_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    modality: Modality
    text: str
    data: bytes
    def __init__(self, model_id: _Optional[str] = ..., modality: _Optional[_Union[Modality, str]] = ..., text: _Optional[str] = ..., data: _Optional[bytes] = ...) -> None: ...

class EncodeQueryResponse(_message.Message):
    __slots__ = ("embedding",)
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    embedding: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, embedding: _Optional[_Iterable[float]] = ...) -> None: ...

class QueryVector(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = ("source_id", "query_vector", "row_key", "k", "filter", "select", "embedding_table")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_VECTOR_FIELD_NUMBER: _ClassVar[int]
    ROW_KEY_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    SELECT_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_TABLE_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    query_vector: QueryVector
    row_key: str
    k: int
    filter: str
    select: _containers.RepeatedScalarFieldContainer[str]
    embedding_table: str
    def __init__(self, source_id: _Optional[str] = ..., query_vector: _Optional[_Union[QueryVector, _Mapping]] = ..., row_key: _Optional[str] = ..., k: _Optional[int] = ..., filter: _Optional[str] = ..., select: _Optional[_Iterable[str]] = ..., embedding_table: _Optional[str] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("hits",)
    HITS_FIELD_NUMBER: _ClassVar[int]
    hits: _containers.RepeatedCompositeFieldContainer[SearchHit]
    def __init__(self, hits: _Optional[_Iterable[_Union[SearchHit, _Mapping]]] = ...) -> None: ...

class SearchHit(_message.Message):
    __slots__ = ("key", "score", "columns")
    class ColumnsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    KEY_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    key: str
    score: float
    columns: _containers.ScalarMap[str, str]
    def __init__(self, key: _Optional[str] = ..., score: _Optional[float] = ..., columns: _Optional[_Mapping[str, str]] = ...) -> None: ...
