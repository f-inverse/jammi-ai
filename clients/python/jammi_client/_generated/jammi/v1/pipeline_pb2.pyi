from . import embedding_pb2 as _embedding_pb2
from . import inference_pb2 as _inference_pb2
from . import trigger_pb2 as _trigger_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Cascade(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CASCADE_UNSPECIFIED: _ClassVar[Cascade]
    CASCADE_REPORT_ONLY: _ClassVar[Cascade]
    CASCADE_DOWNSTREAM: _ClassVar[Cascade]

class AsofDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ASOF_DIRECTION_UNSPECIFIED: _ClassVar[AsofDirection]
    ASOF_DIRECTION_BACKWARD: _ClassVar[AsofDirection]
    ASOF_DIRECTION_FORWARD: _ClassVar[AsofDirection]
    ASOF_DIRECTION_NEAREST: _ClassVar[AsofDirection]

class AsofBoundary(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ASOF_BOUNDARY_UNSPECIFIED: _ClassVar[AsofBoundary]
    ASOF_BOUNDARY_INCLUSIVE: _ClassVar[AsofBoundary]
    ASOF_BOUNDARY_EXCLUSIVE: _ClassVar[AsofBoundary]

class PropagationWeighting(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROPAGATION_WEIGHTING_UNSPECIFIED: _ClassVar[PropagationWeighting]
    PROPAGATION_WEIGHTING_DEGREE_NORMALIZED: _ClassVar[PropagationWeighting]
    PROPAGATION_WEIGHTING_UNIFORM: _ClassVar[PropagationWeighting]
    PROPAGATION_WEIGHTING_EDGE_SIMILARITY: _ClassVar[PropagationWeighting]

class PropagationOutput(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROPAGATION_OUTPUT_UNSPECIFIED: _ClassVar[PropagationOutput]
    PROPAGATION_OUTPUT_FINAL: _ClassVar[PropagationOutput]
    PROPAGATION_OUTPUT_JUMPING_KNOWLEDGE: _ClassVar[PropagationOutput]

class SetAggregator(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SET_AGGREGATOR_UNSPECIFIED: _ClassVar[SetAggregator]
    SET_AGGREGATOR_MEAN: _ClassVar[SetAggregator]
    SET_AGGREGATOR_SUM: _ClassVar[SetAggregator]
    SET_AGGREGATOR_MAX: _ClassVar[SetAggregator]
CASCADE_UNSPECIFIED: Cascade
CASCADE_REPORT_ONLY: Cascade
CASCADE_DOWNSTREAM: Cascade
ASOF_DIRECTION_UNSPECIFIED: AsofDirection
ASOF_DIRECTION_BACKWARD: AsofDirection
ASOF_DIRECTION_FORWARD: AsofDirection
ASOF_DIRECTION_NEAREST: AsofDirection
ASOF_BOUNDARY_UNSPECIFIED: AsofBoundary
ASOF_BOUNDARY_INCLUSIVE: AsofBoundary
ASOF_BOUNDARY_EXCLUSIVE: AsofBoundary
PROPAGATION_WEIGHTING_UNSPECIFIED: PropagationWeighting
PROPAGATION_WEIGHTING_DEGREE_NORMALIZED: PropagationWeighting
PROPAGATION_WEIGHTING_UNIFORM: PropagationWeighting
PROPAGATION_WEIGHTING_EDGE_SIMILARITY: PropagationWeighting
PROPAGATION_OUTPUT_UNSPECIFIED: PropagationOutput
PROPAGATION_OUTPUT_FINAL: PropagationOutput
PROPAGATION_OUTPUT_JUMPING_KNOWLEDGE: PropagationOutput
SET_AGGREGATOR_UNSPECIFIED: SetAggregator
SET_AGGREGATOR_MEAN: SetAggregator
SET_AGGREGATOR_SUM: SetAggregator
SET_AGGREGATOR_MAX: SetAggregator

class RecomputeRequest(_message.Message):
    __slots__ = ("table", "cascade")
    TABLE_FIELD_NUMBER: _ClassVar[int]
    CASCADE_FIELD_NUMBER: _ClassVar[int]
    table: str
    cascade: Cascade
    def __init__(self, table: _Optional[str] = ..., cascade: _Optional[_Union[Cascade, str]] = ...) -> None: ...

class RecomputedTable(_message.Message):
    __slots__ = ("original", "recomputed", "outcome")
    ORIGINAL_FIELD_NUMBER: _ClassVar[int]
    RECOMPUTED_FIELD_NUMBER: _ClassVar[int]
    OUTCOME_FIELD_NUMBER: _ClassVar[int]
    original: str
    recomputed: str
    outcome: _inference_pb2.CacheOutcome
    def __init__(self, original: _Optional[str] = ..., recomputed: _Optional[str] = ..., outcome: _Optional[_Union[_inference_pb2.CacheOutcome, str]] = ...) -> None: ...

class RecomputeReport(_message.Message):
    __slots__ = ("recomputed", "downstream_stale")
    RECOMPUTED_FIELD_NUMBER: _ClassVar[int]
    DOWNSTREAM_STALE_FIELD_NUMBER: _ClassVar[int]
    recomputed: _containers.RepeatedCompositeFieldContainer[RecomputedTable]
    downstream_stale: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, recomputed: _Optional[_Iterable[_Union[RecomputedTable, _Mapping]]] = ..., downstream_stale: _Optional[_Iterable[str]] = ...) -> None: ...

class AsofKey(_message.Message):
    __slots__ = ("by", "time")
    BY_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    by: _containers.RepeatedScalarFieldContainer[str]
    time: str
    def __init__(self, by: _Optional[_Iterable[str]] = ..., time: _Optional[str] = ...) -> None: ...

class AsofTolerance(_message.Message):
    __slots__ = ("duration_micros", "steps")
    DURATION_MICROS_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    duration_micros: int
    steps: int
    def __init__(self, duration_micros: _Optional[int] = ..., steps: _Optional[int] = ...) -> None: ...

class AsofJoinRequest(_message.Message):
    __slots__ = ("spine", "facts", "left", "right", "direction", "boundary", "tolerance", "tie_break_column", "project")
    SPINE_FIELD_NUMBER: _ClassVar[int]
    FACTS_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    TOLERANCE_FIELD_NUMBER: _ClassVar[int]
    TIE_BREAK_COLUMN_FIELD_NUMBER: _ClassVar[int]
    PROJECT_FIELD_NUMBER: _ClassVar[int]
    spine: str
    facts: str
    left: AsofKey
    right: AsofKey
    direction: AsofDirection
    boundary: AsofBoundary
    tolerance: AsofTolerance
    tie_break_column: str
    project: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, spine: _Optional[str] = ..., facts: _Optional[str] = ..., left: _Optional[_Union[AsofKey, _Mapping]] = ..., right: _Optional[_Union[AsofKey, _Mapping]] = ..., direction: _Optional[_Union[AsofDirection, str]] = ..., boundary: _Optional[_Union[AsofBoundary, str]] = ..., tolerance: _Optional[_Union[AsofTolerance, _Mapping]] = ..., tie_break_column: _Optional[str] = ..., project: _Optional[_Iterable[str]] = ...) -> None: ...

class BuildNeighborGraphRequest(_message.Message):
    __slots__ = ("source_id", "k", "min_similarity", "mutual", "exact", "table", "cache")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    MIN_SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    MUTUAL_FIELD_NUMBER: _ClassVar[int]
    EXACT_FIELD_NUMBER: _ClassVar[int]
    TABLE_FIELD_NUMBER: _ClassVar[int]
    CACHE_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    k: int
    min_similarity: float
    mutual: bool
    exact: bool
    table: str
    cache: _inference_pb2.CachePolicy
    def __init__(self, source_id: _Optional[str] = ..., k: _Optional[int] = ..., min_similarity: _Optional[float] = ..., mutual: bool = ..., exact: bool = ..., table: _Optional[str] = ..., cache: _Optional[_Union[_inference_pb2.CachePolicy, str]] = ...) -> None: ...

class PropagateEdgeSource(_message.Message):
    __slots__ = ("edge_source", "src_column", "dst_column", "weight_column")
    EDGE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    SRC_COLUMN_FIELD_NUMBER: _ClassVar[int]
    DST_COLUMN_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_COLUMN_FIELD_NUMBER: _ClassVar[int]
    edge_source: str
    src_column: str
    dst_column: str
    weight_column: str
    def __init__(self, edge_source: _Optional[str] = ..., src_column: _Optional[str] = ..., dst_column: _Optional[str] = ..., weight_column: _Optional[str] = ...) -> None: ...

class PropagateEmbeddingsRequest(_message.Message):
    __slots__ = ("source_id", "embedding_table", "edge_graph_table", "edge_source", "direction", "hops", "weighting", "alpha", "output", "cache")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_TABLE_FIELD_NUMBER: _ClassVar[int]
    EDGE_GRAPH_TABLE_FIELD_NUMBER: _ClassVar[int]
    EDGE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    HOPS_FIELD_NUMBER: _ClassVar[int]
    WEIGHTING_FIELD_NUMBER: _ClassVar[int]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    CACHE_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    embedding_table: str
    edge_graph_table: str
    edge_source: PropagateEdgeSource
    direction: _inference_pb2.EdgeDirection
    hops: int
    weighting: PropagationWeighting
    alpha: float
    output: PropagationOutput
    cache: _inference_pb2.CachePolicy
    def __init__(self, source_id: _Optional[str] = ..., embedding_table: _Optional[str] = ..., edge_graph_table: _Optional[str] = ..., edge_source: _Optional[_Union[PropagateEdgeSource, _Mapping]] = ..., direction: _Optional[_Union[_inference_pb2.EdgeDirection, str]] = ..., hops: _Optional[int] = ..., weighting: _Optional[_Union[PropagationWeighting, str]] = ..., alpha: _Optional[float] = ..., output: _Optional[_Union[PropagationOutput, str]] = ..., cache: _Optional[_Union[_inference_pb2.CachePolicy, str]] = ...) -> None: ...

class AssembleContextRequest(_message.Message):
    __slots__ = ("source_id", "query", "k", "value_columns", "aggregator", "exclude_self", "exclude_key", "split", "edges", "hybrid")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    VALUE_COLUMNS_FIELD_NUMBER: _ClassVar[int]
    AGGREGATOR_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_SELF_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_KEY_FIELD_NUMBER: _ClassVar[int]
    SPLIT_FIELD_NUMBER: _ClassVar[int]
    EDGES_FIELD_NUMBER: _ClassVar[int]
    HYBRID_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    query: _containers.RepeatedScalarFieldContainer[float]
    k: int
    value_columns: _containers.RepeatedScalarFieldContainer[str]
    aggregator: SetAggregator
    exclude_self: bool
    exclude_key: str
    split: str
    edges: _inference_pb2.EdgeGather
    hybrid: bool
    def __init__(self, source_id: _Optional[str] = ..., query: _Optional[_Iterable[float]] = ..., k: _Optional[int] = ..., value_columns: _Optional[_Iterable[str]] = ..., aggregator: _Optional[_Union[SetAggregator, str]] = ..., exclude_self: bool = ..., exclude_key: _Optional[str] = ..., split: _Optional[str] = ..., edges: _Optional[_Union[_inference_pb2.EdgeGather, _Mapping]] = ..., hybrid: bool = ...) -> None: ...

class ContextVector(_message.Message):
    __slots__ = ("values",)
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ...) -> None: ...

class AssembleContextResponse(_message.Message):
    __slots__ = ("context_vector", "context_size", "context_keys", "value_rows", "source")
    CONTEXT_VECTOR_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_SIZE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_KEYS_FIELD_NUMBER: _ClassVar[int]
    VALUE_ROWS_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    context_vector: ContextVector
    context_size: int
    context_keys: _containers.RepeatedScalarFieldContainer[str]
    value_rows: _trigger_pb2.ArrowBatch
    source: str
    def __init__(self, context_vector: _Optional[_Union[ContextVector, _Mapping]] = ..., context_size: _Optional[int] = ..., context_keys: _Optional[_Iterable[str]] = ..., value_rows: _Optional[_Union[_trigger_pb2.ArrowBatch, _Mapping]] = ..., source: _Optional[str] = ...) -> None: ...
