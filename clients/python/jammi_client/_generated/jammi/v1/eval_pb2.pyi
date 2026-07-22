from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EvalTask(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EVAL_TASK_UNSPECIFIED: _ClassVar[EvalTask]
    EVAL_TASK_CLASSIFICATION: _ClassVar[EvalTask]
    EVAL_TASK_NER: _ClassVar[EvalTask]
EVAL_TASK_UNSPECIFIED: EvalTask
EVAL_TASK_CLASSIFICATION: EvalTask
EVAL_TASK_NER: EvalTask

class EvalEmbeddingsRequest(_message.Message):
    __slots__ = ("source_id", "embedding_table", "golden_source", "k", "cohorts", "tenant_id")
    class CohortsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: CohortTags
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[CohortTags, _Mapping]] = ...) -> None: ...
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_TABLE_FIELD_NUMBER: _ClassVar[int]
    GOLDEN_SOURCE_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    COHORTS_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    embedding_table: str
    golden_source: str
    k: int
    cohorts: _containers.MessageMap[str, CohortTags]
    tenant_id: str
    def __init__(self, source_id: _Optional[str] = ..., embedding_table: _Optional[str] = ..., golden_source: _Optional[str] = ..., k: _Optional[int] = ..., cohorts: _Optional[_Mapping[str, CohortTags]] = ..., tenant_id: _Optional[str] = ...) -> None: ...

class CohortTags(_message.Message):
    __slots__ = ("tags",)
    class TagsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TAGS_FIELD_NUMBER: _ClassVar[int]
    tags: _containers.ScalarMap[str, str]
    def __init__(self, tags: _Optional[_Mapping[str, str]] = ...) -> None: ...

class EvalPerQueryRequest(_message.Message):
    __slots__ = ("eval_run_id", "tenant_id")
    EVAL_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    eval_run_id: str
    tenant_id: str
    def __init__(self, eval_run_id: _Optional[str] = ..., tenant_id: _Optional[str] = ...) -> None: ...

class EvalInferenceRequest(_message.Message):
    __slots__ = ("model_id", "source_id", "columns", "task", "golden_source", "label_column", "tenant_id")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    GOLDEN_SOURCE_FIELD_NUMBER: _ClassVar[int]
    LABEL_COLUMN_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    source_id: str
    columns: _containers.RepeatedScalarFieldContainer[str]
    task: EvalTask
    golden_source: str
    label_column: str
    tenant_id: str
    def __init__(self, model_id: _Optional[str] = ..., source_id: _Optional[str] = ..., columns: _Optional[_Iterable[str]] = ..., task: _Optional[_Union[EvalTask, str]] = ..., golden_source: _Optional[str] = ..., label_column: _Optional[str] = ..., tenant_id: _Optional[str] = ...) -> None: ...

class EvalCompareRequest(_message.Message):
    __slots__ = ("embedding_tables", "source_id", "golden_source", "k", "tenant_id")
    EMBEDDING_TABLES_FIELD_NUMBER: _ClassVar[int]
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    GOLDEN_SOURCE_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    embedding_tables: _containers.RepeatedScalarFieldContainer[str]
    source_id: str
    golden_source: str
    k: int
    tenant_id: str
    def __init__(self, embedding_tables: _Optional[_Iterable[str]] = ..., source_id: _Optional[str] = ..., golden_source: _Optional[str] = ..., k: _Optional[int] = ..., tenant_id: _Optional[str] = ...) -> None: ...

class AggregateMetrics(_message.Message):
    __slots__ = ("recall_at_k", "precision_at_k", "mrr", "ndcg")
    RECALL_AT_K_FIELD_NUMBER: _ClassVar[int]
    PRECISION_AT_K_FIELD_NUMBER: _ClassVar[int]
    MRR_FIELD_NUMBER: _ClassVar[int]
    NDCG_FIELD_NUMBER: _ClassVar[int]
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg: float
    def __init__(self, recall_at_k: _Optional[float] = ..., precision_at_k: _Optional[float] = ..., mrr: _Optional[float] = ..., ndcg: _Optional[float] = ...) -> None: ...

class QueryMetrics(_message.Message):
    __slots__ = ("recall", "precision", "mrr", "ndcg")
    RECALL_FIELD_NUMBER: _ClassVar[int]
    PRECISION_FIELD_NUMBER: _ClassVar[int]
    MRR_FIELD_NUMBER: _ClassVar[int]
    NDCG_FIELD_NUMBER: _ClassVar[int]
    recall: float
    precision: float
    mrr: float
    ndcg: float
    def __init__(self, recall: _Optional[float] = ..., precision: _Optional[float] = ..., mrr: _Optional[float] = ..., ndcg: _Optional[float] = ...) -> None: ...

class RecallAtK(_message.Message):
    __slots__ = ("k", "recall")
    K_FIELD_NUMBER: _ClassVar[int]
    RECALL_FIELD_NUMBER: _ClassVar[int]
    k: int
    recall: float
    def __init__(self, k: _Optional[int] = ..., recall: _Optional[float] = ...) -> None: ...

class PerQueryRecord(_message.Message):
    __slots__ = ("query_id", "metrics", "recall_at_ks", "distance", "cohorts")
    class CohortsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    QUERY_ID_FIELD_NUMBER: _ClassVar[int]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    RECALL_AT_KS_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FIELD_NUMBER: _ClassVar[int]
    COHORTS_FIELD_NUMBER: _ClassVar[int]
    query_id: str
    metrics: QueryMetrics
    recall_at_ks: _containers.RepeatedCompositeFieldContainer[RecallAtK]
    distance: float
    cohorts: _containers.ScalarMap[str, str]
    def __init__(self, query_id: _Optional[str] = ..., metrics: _Optional[_Union[QueryMetrics, _Mapping]] = ..., recall_at_ks: _Optional[_Iterable[_Union[RecallAtK, _Mapping]]] = ..., distance: _Optional[float] = ..., cohorts: _Optional[_Mapping[str, str]] = ...) -> None: ...

class EmbeddingEvalReport(_message.Message):
    __slots__ = ("eval_run_id", "aggregate", "per_query")
    EVAL_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    AGGREGATE_FIELD_NUMBER: _ClassVar[int]
    PER_QUERY_FIELD_NUMBER: _ClassVar[int]
    eval_run_id: str
    aggregate: AggregateMetrics
    per_query: _containers.RepeatedCompositeFieldContainer[PerQueryRecord]
    def __init__(self, eval_run_id: _Optional[str] = ..., aggregate: _Optional[_Union[AggregateMetrics, _Mapping]] = ..., per_query: _Optional[_Iterable[_Union[PerQueryRecord, _Mapping]]] = ...) -> None: ...

class PerQueryEvalRecord(_message.Message):
    __slots__ = ("eval_run_id", "query_id", "cohorts_json", "metrics_json")
    EVAL_RUN_ID_FIELD_NUMBER: _ClassVar[int]
    QUERY_ID_FIELD_NUMBER: _ClassVar[int]
    COHORTS_JSON_FIELD_NUMBER: _ClassVar[int]
    METRICS_JSON_FIELD_NUMBER: _ClassVar[int]
    eval_run_id: str
    query_id: str
    cohorts_json: str
    metrics_json: str
    def __init__(self, eval_run_id: _Optional[str] = ..., query_id: _Optional[str] = ..., cohorts_json: _Optional[str] = ..., metrics_json: _Optional[str] = ...) -> None: ...

class EvalPerQueryResponse(_message.Message):
    __slots__ = ("records",)
    RECORDS_FIELD_NUMBER: _ClassVar[int]
    records: _containers.RepeatedCompositeFieldContainer[PerQueryEvalRecord]
    def __init__(self, records: _Optional[_Iterable[_Union[PerQueryEvalRecord, _Mapping]]] = ...) -> None: ...

class ClassMetrics(_message.Message):
    __slots__ = ("precision", "recall", "f1")
    PRECISION_FIELD_NUMBER: _ClassVar[int]
    RECALL_FIELD_NUMBER: _ClassVar[int]
    F1_FIELD_NUMBER: _ClassVar[int]
    precision: float
    recall: float
    f1: float
    def __init__(self, precision: _Optional[float] = ..., recall: _Optional[float] = ..., f1: _Optional[float] = ...) -> None: ...

class ClassificationResult(_message.Message):
    __slots__ = ("accuracy", "f1", "per_class")
    class PerClassEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ClassMetrics
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ClassMetrics, _Mapping]] = ...) -> None: ...
    ACCURACY_FIELD_NUMBER: _ClassVar[int]
    F1_FIELD_NUMBER: _ClassVar[int]
    PER_CLASS_FIELD_NUMBER: _ClassVar[int]
    accuracy: float
    f1: float
    per_class: _containers.MessageMap[str, ClassMetrics]
    def __init__(self, accuracy: _Optional[float] = ..., f1: _Optional[float] = ..., per_class: _Optional[_Mapping[str, ClassMetrics]] = ...) -> None: ...

class TypeMetrics(_message.Message):
    __slots__ = ("precision", "recall", "f1", "support")
    PRECISION_FIELD_NUMBER: _ClassVar[int]
    RECALL_FIELD_NUMBER: _ClassVar[int]
    F1_FIELD_NUMBER: _ClassVar[int]
    SUPPORT_FIELD_NUMBER: _ClassVar[int]
    precision: float
    recall: float
    f1: float
    support: int
    def __init__(self, precision: _Optional[float] = ..., recall: _Optional[float] = ..., f1: _Optional[float] = ..., support: _Optional[int] = ...) -> None: ...

class NerMetrics(_message.Message):
    __slots__ = ("precision", "recall", "f1", "per_type")
    class PerTypeEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: TypeMetrics
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[TypeMetrics, _Mapping]] = ...) -> None: ...
    PRECISION_FIELD_NUMBER: _ClassVar[int]
    RECALL_FIELD_NUMBER: _ClassVar[int]
    F1_FIELD_NUMBER: _ClassVar[int]
    PER_TYPE_FIELD_NUMBER: _ClassVar[int]
    precision: float
    recall: float
    f1: float
    per_type: _containers.MessageMap[str, TypeMetrics]
    def __init__(self, precision: _Optional[float] = ..., recall: _Optional[float] = ..., f1: _Optional[float] = ..., per_type: _Optional[_Mapping[str, TypeMetrics]] = ...) -> None: ...

class Entity(_message.Message):
    __slots__ = ("label", "start", "end", "text", "confidence")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    label: str
    start: int
    end: int
    text: str
    confidence: float
    def __init__(self, label: _Optional[str] = ..., start: _Optional[int] = ..., end: _Optional[int] = ..., text: _Optional[str] = ..., confidence: _Optional[float] = ...) -> None: ...

class InferenceAggregate(_message.Message):
    __slots__ = ("classification", "ner")
    CLASSIFICATION_FIELD_NUMBER: _ClassVar[int]
    NER_FIELD_NUMBER: _ClassVar[int]
    classification: ClassificationResult
    ner: NerMetrics
    def __init__(self, classification: _Optional[_Union[ClassificationResult, _Mapping]] = ..., ner: _Optional[_Union[NerMetrics, _Mapping]] = ...) -> None: ...

class PerRecordPrediction(_message.Message):
    __slots__ = ("classification", "ner")
    class Classification(_message.Message):
        __slots__ = ("record_id", "predicted", "gold")
        RECORD_ID_FIELD_NUMBER: _ClassVar[int]
        PREDICTED_FIELD_NUMBER: _ClassVar[int]
        GOLD_FIELD_NUMBER: _ClassVar[int]
        record_id: str
        predicted: str
        gold: str
        def __init__(self, record_id: _Optional[str] = ..., predicted: _Optional[str] = ..., gold: _Optional[str] = ...) -> None: ...
    class Ner(_message.Message):
        __slots__ = ("record_id", "predicted", "gold")
        RECORD_ID_FIELD_NUMBER: _ClassVar[int]
        PREDICTED_FIELD_NUMBER: _ClassVar[int]
        GOLD_FIELD_NUMBER: _ClassVar[int]
        record_id: str
        predicted: _containers.RepeatedCompositeFieldContainer[Entity]
        gold: _containers.RepeatedCompositeFieldContainer[Entity]
        def __init__(self, record_id: _Optional[str] = ..., predicted: _Optional[_Iterable[_Union[Entity, _Mapping]]] = ..., gold: _Optional[_Iterable[_Union[Entity, _Mapping]]] = ...) -> None: ...
    CLASSIFICATION_FIELD_NUMBER: _ClassVar[int]
    NER_FIELD_NUMBER: _ClassVar[int]
    classification: PerRecordPrediction.Classification
    ner: PerRecordPrediction.Ner
    def __init__(self, classification: _Optional[_Union[PerRecordPrediction.Classification, _Mapping]] = ..., ner: _Optional[_Union[PerRecordPrediction.Ner, _Mapping]] = ...) -> None: ...

class InferenceEvalReport(_message.Message):
    __slots__ = ("aggregate", "per_record")
    AGGREGATE_FIELD_NUMBER: _ClassVar[int]
    PER_RECORD_FIELD_NUMBER: _ClassVar[int]
    aggregate: InferenceAggregate
    per_record: _containers.RepeatedCompositeFieldContainer[PerRecordPrediction]
    def __init__(self, aggregate: _Optional[_Union[InferenceAggregate, _Mapping]] = ..., per_record: _Optional[_Iterable[_Union[PerRecordPrediction, _Mapping]]] = ...) -> None: ...

class MetricDelta(_message.Message):
    __slots__ = ("absolute", "relative")
    ABSOLUTE_FIELD_NUMBER: _ClassVar[int]
    RELATIVE_FIELD_NUMBER: _ClassVar[int]
    absolute: float
    relative: float
    def __init__(self, absolute: _Optional[float] = ..., relative: _Optional[float] = ...) -> None: ...

class AggregateDelta(_message.Message):
    __slots__ = ("recall_at_k", "precision_at_k", "mrr", "ndcg")
    RECALL_AT_K_FIELD_NUMBER: _ClassVar[int]
    PRECISION_AT_K_FIELD_NUMBER: _ClassVar[int]
    MRR_FIELD_NUMBER: _ClassVar[int]
    NDCG_FIELD_NUMBER: _ClassVar[int]
    recall_at_k: MetricDelta
    precision_at_k: MetricDelta
    mrr: MetricDelta
    ndcg: MetricDelta
    def __init__(self, recall_at_k: _Optional[_Union[MetricDelta, _Mapping]] = ..., precision_at_k: _Optional[_Union[MetricDelta, _Mapping]] = ..., mrr: _Optional[_Union[MetricDelta, _Mapping]] = ..., ndcg: _Optional[_Union[MetricDelta, _Mapping]] = ...) -> None: ...

class TableEvalReport(_message.Message):
    __slots__ = ("table_name", "embedding_eval", "delta")
    TABLE_NAME_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_EVAL_FIELD_NUMBER: _ClassVar[int]
    DELTA_FIELD_NUMBER: _ClassVar[int]
    table_name: str
    embedding_eval: EmbeddingEvalReport
    delta: AggregateDelta
    def __init__(self, table_name: _Optional[str] = ..., embedding_eval: _Optional[_Union[EmbeddingEvalReport, _Mapping]] = ..., delta: _Optional[_Union[AggregateDelta, _Mapping]] = ...) -> None: ...

class CompareEvalReport(_message.Message):
    __slots__ = ("per_table",)
    PER_TABLE_FIELD_NUMBER: _ClassVar[int]
    per_table: _containers.RepeatedCompositeFieldContainer[TableEvalReport]
    def __init__(self, per_table: _Optional[_Iterable[_Union[TableEvalReport, _Mapping]]] = ...) -> None: ...
