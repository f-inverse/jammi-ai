from . import inference_pb2 as _inference_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FineTuneMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FINE_TUNE_METHOD_UNSPECIFIED: _ClassVar[FineTuneMethod]
    LORA: _ClassVar[FineTuneMethod]

class ClassificationLoss(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CLASSIFICATION_LOSS_UNSPECIFIED: _ClassVar[ClassificationLoss]
    CROSS_ENTROPY: _ClassVar[ClassificationLoss]

class EarlyStoppingMetric(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EARLY_STOPPING_METRIC_UNSPECIFIED: _ClassVar[EarlyStoppingMetric]
    VAL_LOSS: _ClassVar[EarlyStoppingMetric]
    TRAIN_LOSS: _ClassVar[EarlyStoppingMetric]

class LrSchedule(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LR_SCHEDULE_UNSPECIFIED: _ClassVar[LrSchedule]
    CONSTANT: _ClassVar[LrSchedule]
    COSINE_DECAY: _ClassVar[LrSchedule]
    LINEAR_DECAY: _ClassVar[LrSchedule]

class LoraInitMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LORA_INIT_MODE_UNSPECIFIED: _ClassVar[LoraInitMode]
    ZEROS_B: _ClassVar[LoraInitMode]
    GAUSSIAN: _ClassVar[LoraInitMode]

class BackboneDtype(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BACKBONE_DTYPE_UNSPECIFIED: _ClassVar[BackboneDtype]
    F32: _ClassVar[BackboneDtype]
    BF16: _ClassVar[BackboneDtype]
    F16: _ClassVar[BackboneDtype]
FINE_TUNE_METHOD_UNSPECIFIED: FineTuneMethod
LORA: FineTuneMethod
CLASSIFICATION_LOSS_UNSPECIFIED: ClassificationLoss
CROSS_ENTROPY: ClassificationLoss
EARLY_STOPPING_METRIC_UNSPECIFIED: EarlyStoppingMetric
VAL_LOSS: EarlyStoppingMetric
TRAIN_LOSS: EarlyStoppingMetric
LR_SCHEDULE_UNSPECIFIED: LrSchedule
CONSTANT: LrSchedule
COSINE_DECAY: LrSchedule
LINEAR_DECAY: LrSchedule
LORA_INIT_MODE_UNSPECIFIED: LoraInitMode
ZEROS_B: LoraInitMode
GAUSSIAN: LoraInitMode
BACKBONE_DTYPE_UNSPECIFIED: BackboneDtype
F32: BackboneDtype
BF16: BackboneDtype
F16: BackboneDtype

class EmbeddingLoss(_message.Message):
    __slots__ = ("co_sent", "triplet", "multiple_negatives_ranking")
    class CoSent(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class Triplet(_message.Message):
        __slots__ = ("margin",)
        MARGIN_FIELD_NUMBER: _ClassVar[int]
        margin: float
        def __init__(self, margin: _Optional[float] = ...) -> None: ...
    class MultipleNegativesRanking(_message.Message):
        __slots__ = ("temperature",)
        TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
        temperature: float
        def __init__(self, temperature: _Optional[float] = ...) -> None: ...
    CO_SENT_FIELD_NUMBER: _ClassVar[int]
    TRIPLET_FIELD_NUMBER: _ClassVar[int]
    MULTIPLE_NEGATIVES_RANKING_FIELD_NUMBER: _ClassVar[int]
    co_sent: EmbeddingLoss.CoSent
    triplet: EmbeddingLoss.Triplet
    multiple_negatives_ranking: EmbeddingLoss.MultipleNegativesRanking
    def __init__(self, co_sent: _Optional[_Union[EmbeddingLoss.CoSent, _Mapping]] = ..., triplet: _Optional[_Union[EmbeddingLoss.Triplet, _Mapping]] = ..., multiple_negatives_ranking: _Optional[_Union[EmbeddingLoss.MultipleNegativesRanking, _Mapping]] = ...) -> None: ...

class LayersToTransform(_message.Message):
    __slots__ = ("layers",)
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    layers: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, layers: _Optional[_Iterable[int]] = ...) -> None: ...

class FineTuneConfig(_message.Message):
    __slots__ = ("lora_rank", "lora_alpha", "lora_dropout", "learning_rate", "epochs", "batch_size", "max_seq_length", "embedding_loss", "classification_loss", "gradient_accumulation_steps", "validation_fraction", "early_stopping_patience", "warmup_steps", "lr_schedule", "early_stopping_metric", "target_modules", "layers_to_transform", "use_rslora", "rank_pattern", "init_lora_weights", "backbone_dtype", "weight_decay", "max_grad_norm")
    class RankPatternEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    LORA_RANK_FIELD_NUMBER: _ClassVar[int]
    LORA_ALPHA_FIELD_NUMBER: _ClassVar[int]
    LORA_DROPOUT_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    EPOCHS_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAX_SEQ_LENGTH_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_LOSS_FIELD_NUMBER: _ClassVar[int]
    CLASSIFICATION_LOSS_FIELD_NUMBER: _ClassVar[int]
    GRADIENT_ACCUMULATION_STEPS_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_FRACTION_FIELD_NUMBER: _ClassVar[int]
    EARLY_STOPPING_PATIENCE_FIELD_NUMBER: _ClassVar[int]
    WARMUP_STEPS_FIELD_NUMBER: _ClassVar[int]
    LR_SCHEDULE_FIELD_NUMBER: _ClassVar[int]
    EARLY_STOPPING_METRIC_FIELD_NUMBER: _ClassVar[int]
    TARGET_MODULES_FIELD_NUMBER: _ClassVar[int]
    LAYERS_TO_TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    USE_RSLORA_FIELD_NUMBER: _ClassVar[int]
    RANK_PATTERN_FIELD_NUMBER: _ClassVar[int]
    INIT_LORA_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    BACKBONE_DTYPE_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_DECAY_FIELD_NUMBER: _ClassVar[int]
    MAX_GRAD_NORM_FIELD_NUMBER: _ClassVar[int]
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    learning_rate: float
    epochs: int
    batch_size: int
    max_seq_length: int
    embedding_loss: EmbeddingLoss
    classification_loss: ClassificationLoss
    gradient_accumulation_steps: int
    validation_fraction: float
    early_stopping_patience: int
    warmup_steps: int
    lr_schedule: LrSchedule
    early_stopping_metric: EarlyStoppingMetric
    target_modules: _containers.RepeatedScalarFieldContainer[str]
    layers_to_transform: LayersToTransform
    use_rslora: bool
    rank_pattern: _containers.ScalarMap[str, int]
    init_lora_weights: LoraInitMode
    backbone_dtype: BackboneDtype
    weight_decay: float
    max_grad_norm: float
    def __init__(self, lora_rank: _Optional[int] = ..., lora_alpha: _Optional[float] = ..., lora_dropout: _Optional[float] = ..., learning_rate: _Optional[float] = ..., epochs: _Optional[int] = ..., batch_size: _Optional[int] = ..., max_seq_length: _Optional[int] = ..., embedding_loss: _Optional[_Union[EmbeddingLoss, _Mapping]] = ..., classification_loss: _Optional[_Union[ClassificationLoss, str]] = ..., gradient_accumulation_steps: _Optional[int] = ..., validation_fraction: _Optional[float] = ..., early_stopping_patience: _Optional[int] = ..., warmup_steps: _Optional[int] = ..., lr_schedule: _Optional[_Union[LrSchedule, str]] = ..., early_stopping_metric: _Optional[_Union[EarlyStoppingMetric, str]] = ..., target_modules: _Optional[_Iterable[str]] = ..., layers_to_transform: _Optional[_Union[LayersToTransform, _Mapping]] = ..., use_rslora: _Optional[bool] = ..., rank_pattern: _Optional[_Mapping[str, int]] = ..., init_lora_weights: _Optional[_Union[LoraInitMode, str]] = ..., backbone_dtype: _Optional[_Union[BackboneDtype, str]] = ..., weight_decay: _Optional[float] = ..., max_grad_norm: _Optional[float] = ...) -> None: ...

class StartFineTuneRequest(_message.Message):
    __slots__ = ("source_id", "base_model", "columns", "method", "task", "config")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    base_model: str
    columns: _containers.RepeatedScalarFieldContainer[str]
    method: FineTuneMethod
    task: _inference_pb2.ModelTask
    config: FineTuneConfig
    def __init__(self, source_id: _Optional[str] = ..., base_model: _Optional[str] = ..., columns: _Optional[_Iterable[str]] = ..., method: _Optional[_Union[FineTuneMethod, str]] = ..., task: _Optional[_Union[_inference_pb2.ModelTask, str]] = ..., config: _Optional[_Union[FineTuneConfig, _Mapping]] = ...) -> None: ...

class StartFineTuneResponse(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class FineTuneStatusRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class FineTuneStatusResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...
