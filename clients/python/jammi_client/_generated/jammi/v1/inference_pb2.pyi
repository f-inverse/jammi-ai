from . import trigger_pb2 as _trigger_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CachePolicy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CACHE_POLICY_UNSPECIFIED: _ClassVar[CachePolicy]
    CACHE_POLICY_USE: _ClassVar[CachePolicy]
    CACHE_POLICY_BYPASS: _ClassVar[CachePolicy]

class CacheOutcome(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CACHE_OUTCOME_UNSPECIFIED: _ClassVar[CacheOutcome]
    CACHE_OUTCOME_COMPUTED: _ClassVar[CacheOutcome]
    CACHE_OUTCOME_REUSED: _ClassVar[CacheOutcome]

class ModelTask(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_TASK_UNSPECIFIED: _ClassVar[ModelTask]
    TEXT_EMBEDDING: _ClassVar[ModelTask]
    IMAGE_EMBEDDING: _ClassVar[ModelTask]
    AUDIO_EMBEDDING: _ClassVar[ModelTask]
    CLASSIFICATION: _ClassVar[ModelTask]
    NER: _ClassVar[ModelTask]
    REGRESSION: _ClassVar[ModelTask]

class EdgeDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EDGE_DIRECTION_UNSPECIFIED: _ClassVar[EdgeDirection]
    OUT: _ClassVar[EdgeDirection]
    IN: _ClassVar[EdgeDirection]
    UNDIRECTED: _ClassVar[EdgeDirection]
CACHE_POLICY_UNSPECIFIED: CachePolicy
CACHE_POLICY_USE: CachePolicy
CACHE_POLICY_BYPASS: CachePolicy
CACHE_OUTCOME_UNSPECIFIED: CacheOutcome
CACHE_OUTCOME_COMPUTED: CacheOutcome
CACHE_OUTCOME_REUSED: CacheOutcome
MODEL_TASK_UNSPECIFIED: ModelTask
TEXT_EMBEDDING: ModelTask
IMAGE_EMBEDDING: ModelTask
AUDIO_EMBEDDING: ModelTask
CLASSIFICATION: ModelTask
NER: ModelTask
REGRESSION: ModelTask
EDGE_DIRECTION_UNSPECIFIED: EdgeDirection
OUT: EdgeDirection
IN: EdgeDirection
UNDIRECTED: EdgeDirection

class InferRequest(_message.Message):
    __slots__ = ("source_id", "model_id", "task", "columns", "key_column", "tenant_id", "cache")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    KEY_COLUMN_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    CACHE_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    model_id: str
    task: ModelTask
    columns: _containers.RepeatedScalarFieldContainer[str]
    key_column: str
    tenant_id: str
    cache: CachePolicy
    def __init__(self, source_id: _Optional[str] = ..., model_id: _Optional[str] = ..., task: _Optional[_Union[ModelTask, str]] = ..., columns: _Optional[_Iterable[str]] = ..., key_column: _Optional[str] = ..., tenant_id: _Optional[str] = ..., cache: _Optional[_Union[CachePolicy, str]] = ...) -> None: ...

class InferResponse(_message.Message):
    __slots__ = ("result", "cache_outcome")
    RESULT_FIELD_NUMBER: _ClassVar[int]
    CACHE_OUTCOME_FIELD_NUMBER: _ClassVar[int]
    result: _trigger_pb2.ArrowBatch
    cache_outcome: CacheOutcome
    def __init__(self, result: _Optional[_Union[_trigger_pb2.ArrowBatch, _Mapping]] = ..., cache_outcome: _Optional[_Union[CacheOutcome, str]] = ...) -> None: ...

class EdgeGather(_message.Message):
    __slots__ = ("edge_source", "src_column", "dst_column", "type_column", "weight_column", "hops", "fanout", "direction", "edge_types", "min_weight")
    EDGE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    SRC_COLUMN_FIELD_NUMBER: _ClassVar[int]
    DST_COLUMN_FIELD_NUMBER: _ClassVar[int]
    TYPE_COLUMN_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_COLUMN_FIELD_NUMBER: _ClassVar[int]
    HOPS_FIELD_NUMBER: _ClassVar[int]
    FANOUT_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    EDGE_TYPES_FIELD_NUMBER: _ClassVar[int]
    MIN_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    edge_source: str
    src_column: str
    dst_column: str
    type_column: str
    weight_column: str
    hops: int
    fanout: int
    direction: EdgeDirection
    edge_types: _containers.RepeatedScalarFieldContainer[str]
    min_weight: float
    def __init__(self, edge_source: _Optional[str] = ..., src_column: _Optional[str] = ..., dst_column: _Optional[str] = ..., type_column: _Optional[str] = ..., weight_column: _Optional[str] = ..., hops: _Optional[int] = ..., fanout: _Optional[int] = ..., direction: _Optional[_Union[EdgeDirection, str]] = ..., edge_types: _Optional[_Iterable[str]] = ..., min_weight: _Optional[float] = ...) -> None: ...

class PredictRequest(_message.Message):
    __slots__ = ("model_id", "source", "target_key", "split", "edges", "hybrid_ann_k")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    TARGET_KEY_FIELD_NUMBER: _ClassVar[int]
    SPLIT_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    HYBRID_ANN_K_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    source: str
    target_key: str
    split: str
    edges: EdgeGather
    hybrid_ann_k: int
    def __init__(self, model_id: _Optional[str] = ..., source: _Optional[str] = ..., target_key: _Optional[str] = ..., split: _Optional[str] = ..., edges: _Optional[_Union[EdgeGather, _Mapping]] = ..., hybrid_ann_k: _Optional[int] = ...) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ("gaussian", "quantile", "source", "context_ref")
    class Gaussian(_message.Message):
        __slots__ = ("mean", "std")
        MEAN_FIELD_NUMBER: _ClassVar[int]
        STD_FIELD_NUMBER: _ClassVar[int]
        mean: float
        std: float
        def __init__(self, mean: _Optional[float] = ..., std: _Optional[float] = ...) -> None: ...
    class Quantile(_message.Message):
        __slots__ = ("points",)
        POINTS_FIELD_NUMBER: _ClassVar[int]
        points: _containers.RepeatedCompositeFieldContainer[PredictResponse.QuantilePoint]
        def __init__(self, points: _Optional[_Iterable[_Union[PredictResponse.QuantilePoint, _Mapping]]] = ...) -> None: ...
    class QuantilePoint(_message.Message):
        __slots__ = ("level", "value")
        LEVEL_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        level: float
        value: float
        def __init__(self, level: _Optional[float] = ..., value: _Optional[float] = ...) -> None: ...
    GAUSSIAN_FIELD_NUMBER: _ClassVar[int]
    QUANTILE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_REF_FIELD_NUMBER: _ClassVar[int]
    gaussian: PredictResponse.Gaussian
    quantile: PredictResponse.Quantile
    source: str
    context_ref: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, gaussian: _Optional[_Union[PredictResponse.Gaussian, _Mapping]] = ..., quantile: _Optional[_Union[PredictResponse.Quantile, _Mapping]] = ..., source: _Optional[str] = ..., context_ref: _Optional[_Iterable[str]] = ...) -> None: ...
