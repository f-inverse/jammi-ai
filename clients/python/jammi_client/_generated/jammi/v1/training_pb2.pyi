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

class EdgeProvenance(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EDGE_PROVENANCE_UNSPECIFIED: _ClassVar[EdgeProvenance]
    DECLARED: _ClassVar[EdgeProvenance]
    SIMILARITY: _ClassVar[EdgeProvenance]

class ContextArchitecture(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONTEXT_ARCHITECTURE_UNSPECIFIED: _ClassVar[ContextArchitecture]
    CNP: _ClassVar[ContextArchitecture]
    ATTN_CNP: _ClassVar[ContextArchitecture]
    TNP: _ClassVar[ContextArchitecture]
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
EDGE_PROVENANCE_UNSPECIFIED: EdgeProvenance
DECLARED: EdgeProvenance
SIMILARITY: EdgeProvenance
CONTEXT_ARCHITECTURE_UNSPECIFIED: ContextArchitecture
CNP: ContextArchitecture
ATTN_CNP: ContextArchitecture
TNP: ContextArchitecture

class EmbeddingLoss(_message.Message):
    __slots__ = ("co_sent", "triplet", "multiple_negatives_ranking", "angle", "cosine_mse")
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
    class AnglE(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class CosineMse(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    CO_SENT_FIELD_NUMBER: _ClassVar[int]
    TRIPLET_FIELD_NUMBER: _ClassVar[int]
    MULTIPLE_NEGATIVES_RANKING_FIELD_NUMBER: _ClassVar[int]
    ANGLE_FIELD_NUMBER: _ClassVar[int]
    COSINE_MSE_FIELD_NUMBER: _ClassVar[int]
    co_sent: EmbeddingLoss.CoSent
    triplet: EmbeddingLoss.Triplet
    multiple_negatives_ranking: EmbeddingLoss.MultipleNegativesRanking
    angle: EmbeddingLoss.AnglE
    cosine_mse: EmbeddingLoss.CosineMse
    def __init__(self, co_sent: _Optional[_Union[EmbeddingLoss.CoSent, _Mapping]] = ..., triplet: _Optional[_Union[EmbeddingLoss.Triplet, _Mapping]] = ..., multiple_negatives_ranking: _Optional[_Union[EmbeddingLoss.MultipleNegativesRanking, _Mapping]] = ..., angle: _Optional[_Union[EmbeddingLoss.AnglE, _Mapping]] = ..., cosine_mse: _Optional[_Union[EmbeddingLoss.CosineMse, _Mapping]] = ...) -> None: ...

class RegressionLoss(_message.Message):
    __slots__ = ("gaussian_nll", "beta_nll", "crps", "pinball")
    class GaussianNll(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class BetaNll(_message.Message):
        __slots__ = ("beta",)
        BETA_FIELD_NUMBER: _ClassVar[int]
        beta: float
        def __init__(self, beta: _Optional[float] = ...) -> None: ...
    class Crps(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class Pinball(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    GAUSSIAN_NLL_FIELD_NUMBER: _ClassVar[int]
    BETA_NLL_FIELD_NUMBER: _ClassVar[int]
    CRPS_FIELD_NUMBER: _ClassVar[int]
    PINBALL_FIELD_NUMBER: _ClassVar[int]
    gaussian_nll: RegressionLoss.GaussianNll
    beta_nll: RegressionLoss.BetaNll
    crps: RegressionLoss.Crps
    pinball: RegressionLoss.Pinball
    def __init__(self, gaussian_nll: _Optional[_Union[RegressionLoss.GaussianNll, _Mapping]] = ..., beta_nll: _Optional[_Union[RegressionLoss.BetaNll, _Mapping]] = ..., crps: _Optional[_Union[RegressionLoss.Crps, _Mapping]] = ..., pinball: _Optional[_Union[RegressionLoss.Pinball, _Mapping]] = ...) -> None: ...

class LayersToTransform(_message.Message):
    __slots__ = ("layers",)
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    layers: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, layers: _Optional[_Iterable[int]] = ...) -> None: ...

class HardNegativeConfig(_message.Message):
    __slots__ = ("mine", "k", "exclude_hops", "refresh_every")
    MINE_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_HOPS_FIELD_NUMBER: _ClassVar[int]
    REFRESH_EVERY_FIELD_NUMBER: _ClassVar[int]
    mine: bool
    k: int
    exclude_hops: int
    refresh_every: int
    def __init__(self, mine: bool = ..., k: _Optional[int] = ..., exclude_hops: _Optional[int] = ..., refresh_every: _Optional[int] = ...) -> None: ...

class FineTuneConfig(_message.Message):
    __slots__ = ("lora_rank", "lora_alpha", "lora_dropout", "learning_rate", "epochs", "batch_size", "max_seq_length", "embedding_loss", "classification_loss", "gradient_accumulation_steps", "validation_fraction", "early_stopping_patience", "warmup_steps", "lr_schedule", "early_stopping_metric", "target_modules", "layers_to_transform", "use_rslora", "rank_pattern", "init_lora_weights", "backbone_dtype", "weight_decay", "max_grad_norm", "cached", "hard_negatives", "matryoshka_dims", "regression_loss", "quantile_levels", "seed")
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
    CACHED_FIELD_NUMBER: _ClassVar[int]
    HARD_NEGATIVES_FIELD_NUMBER: _ClassVar[int]
    MATRYOSHKA_DIMS_FIELD_NUMBER: _ClassVar[int]
    REGRESSION_LOSS_FIELD_NUMBER: _ClassVar[int]
    QUANTILE_LEVELS_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
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
    cached: bool
    hard_negatives: HardNegativeConfig
    matryoshka_dims: _containers.RepeatedScalarFieldContainer[int]
    regression_loss: RegressionLoss
    quantile_levels: _containers.RepeatedScalarFieldContainer[float]
    seed: int
    def __init__(self, lora_rank: _Optional[int] = ..., lora_alpha: _Optional[float] = ..., lora_dropout: _Optional[float] = ..., learning_rate: _Optional[float] = ..., epochs: _Optional[int] = ..., batch_size: _Optional[int] = ..., max_seq_length: _Optional[int] = ..., embedding_loss: _Optional[_Union[EmbeddingLoss, _Mapping]] = ..., classification_loss: _Optional[_Union[ClassificationLoss, str]] = ..., gradient_accumulation_steps: _Optional[int] = ..., validation_fraction: _Optional[float] = ..., early_stopping_patience: _Optional[int] = ..., warmup_steps: _Optional[int] = ..., lr_schedule: _Optional[_Union[LrSchedule, str]] = ..., early_stopping_metric: _Optional[_Union[EarlyStoppingMetric, str]] = ..., target_modules: _Optional[_Iterable[str]] = ..., layers_to_transform: _Optional[_Union[LayersToTransform, _Mapping]] = ..., use_rslora: bool = ..., rank_pattern: _Optional[_Mapping[str, int]] = ..., init_lora_weights: _Optional[_Union[LoraInitMode, str]] = ..., backbone_dtype: _Optional[_Union[BackboneDtype, str]] = ..., weight_decay: _Optional[float] = ..., max_grad_norm: _Optional[float] = ..., cached: bool = ..., hard_negatives: _Optional[_Union[HardNegativeConfig, _Mapping]] = ..., matryoshka_dims: _Optional[_Iterable[int]] = ..., regression_loss: _Optional[_Union[RegressionLoss, _Mapping]] = ..., quantile_levels: _Optional[_Iterable[float]] = ..., seed: _Optional[int] = ...) -> None: ...

class GraphFineTuneSources(_message.Message):
    __slots__ = ("node_source", "id_column", "text_column", "edge_source", "src_column", "dst_column", "provenance")
    NODE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    ID_COLUMN_FIELD_NUMBER: _ClassVar[int]
    TEXT_COLUMN_FIELD_NUMBER: _ClassVar[int]
    EDGE_SOURCE_FIELD_NUMBER: _ClassVar[int]
    SRC_COLUMN_FIELD_NUMBER: _ClassVar[int]
    DST_COLUMN_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    node_source: str
    id_column: str
    text_column: str
    edge_source: str
    src_column: str
    dst_column: str
    provenance: EdgeProvenance
    def __init__(self, node_source: _Optional[str] = ..., id_column: _Optional[str] = ..., text_column: _Optional[str] = ..., edge_source: _Optional[str] = ..., src_column: _Optional[str] = ..., dst_column: _Optional[str] = ..., provenance: _Optional[_Union[EdgeProvenance, str]] = ...) -> None: ...

class GraphSampleConfig(_message.Message):
    __slots__ = ("walk_length", "walks_per_node", "return_p", "in_out_q", "hard_negatives", "exclude_hops", "min_negatives", "seed")
    WALK_LENGTH_FIELD_NUMBER: _ClassVar[int]
    WALKS_PER_NODE_FIELD_NUMBER: _ClassVar[int]
    RETURN_P_FIELD_NUMBER: _ClassVar[int]
    IN_OUT_Q_FIELD_NUMBER: _ClassVar[int]
    HARD_NEGATIVES_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_HOPS_FIELD_NUMBER: _ClassVar[int]
    MIN_NEGATIVES_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    walk_length: int
    walks_per_node: int
    return_p: float
    in_out_q: float
    hard_negatives: int
    exclude_hops: int
    min_negatives: int
    seed: int
    def __init__(self, walk_length: _Optional[int] = ..., walks_per_node: _Optional[int] = ..., return_p: _Optional[float] = ..., in_out_q: _Optional[float] = ..., hard_negatives: _Optional[int] = ..., exclude_hops: _Optional[int] = ..., min_negatives: _Optional[int] = ..., seed: _Optional[int] = ...) -> None: ...

class PredictiveHead(_message.Message):
    __slots__ = ("gaussian", "quantile")
    class Gaussian(_message.Message):
        __slots__ = ("objective",)
        OBJECTIVE_FIELD_NUMBER: _ClassVar[int]
        objective: GaussianObjective
        def __init__(self, objective: _Optional[_Union[GaussianObjective, _Mapping]] = ...) -> None: ...
    class Quantile(_message.Message):
        __slots__ = ("levels",)
        LEVELS_FIELD_NUMBER: _ClassVar[int]
        levels: _containers.RepeatedScalarFieldContainer[float]
        def __init__(self, levels: _Optional[_Iterable[float]] = ...) -> None: ...
    GAUSSIAN_FIELD_NUMBER: _ClassVar[int]
    QUANTILE_FIELD_NUMBER: _ClassVar[int]
    gaussian: PredictiveHead.Gaussian
    quantile: PredictiveHead.Quantile
    def __init__(self, gaussian: _Optional[_Union[PredictiveHead.Gaussian, _Mapping]] = ..., quantile: _Optional[_Union[PredictiveHead.Quantile, _Mapping]] = ...) -> None: ...

class GaussianObjective(_message.Message):
    __slots__ = ("nll", "crps")
    class Nll(_message.Message):
        __slots__ = ("beta",)
        BETA_FIELD_NUMBER: _ClassVar[int]
        beta: float
        def __init__(self, beta: _Optional[float] = ...) -> None: ...
    class Crps(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    NLL_FIELD_NUMBER: _ClassVar[int]
    CRPS_FIELD_NUMBER: _ClassVar[int]
    nll: GaussianObjective.Nll
    crps: GaussianObjective.Crps
    def __init__(self, nll: _Optional[_Union[GaussianObjective.Nll, _Mapping]] = ..., crps: _Optional[_Union[GaussianObjective.Crps, _Mapping]] = ...) -> None: ...

class ContextPredictorTrainConfig(_message.Message):
    __slots__ = ("model_id", "architecture", "key_column", "task_column", "value_column", "context_k", "hidden_dim", "num_heads", "num_layers", "head", "epochs", "learning_rate", "grad_clip", "test_task_fraction", "min_task_count", "seed")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    ARCHITECTURE_FIELD_NUMBER: _ClassVar[int]
    KEY_COLUMN_FIELD_NUMBER: _ClassVar[int]
    TASK_COLUMN_FIELD_NUMBER: _ClassVar[int]
    VALUE_COLUMN_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_K_FIELD_NUMBER: _ClassVar[int]
    HIDDEN_DIM_FIELD_NUMBER: _ClassVar[int]
    NUM_HEADS_FIELD_NUMBER: _ClassVar[int]
    NUM_LAYERS_FIELD_NUMBER: _ClassVar[int]
    HEAD_FIELD_NUMBER: _ClassVar[int]
    EPOCHS_FIELD_NUMBER: _ClassVar[int]
    LEARNING_RATE_FIELD_NUMBER: _ClassVar[int]
    GRAD_CLIP_FIELD_NUMBER: _ClassVar[int]
    TEST_TASK_FRACTION_FIELD_NUMBER: _ClassVar[int]
    MIN_TASK_COUNT_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    architecture: ContextArchitecture
    key_column: str
    task_column: str
    value_column: str
    context_k: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    head: PredictiveHead
    epochs: int
    learning_rate: float
    grad_clip: float
    test_task_fraction: float
    min_task_count: int
    seed: int
    def __init__(self, model_id: _Optional[str] = ..., architecture: _Optional[_Union[ContextArchitecture, str]] = ..., key_column: _Optional[str] = ..., task_column: _Optional[str] = ..., value_column: _Optional[str] = ..., context_k: _Optional[int] = ..., hidden_dim: _Optional[int] = ..., num_heads: _Optional[int] = ..., num_layers: _Optional[int] = ..., head: _Optional[_Union[PredictiveHead, _Mapping]] = ..., epochs: _Optional[int] = ..., learning_rate: _Optional[float] = ..., grad_clip: _Optional[float] = ..., test_task_fraction: _Optional[float] = ..., min_task_count: _Optional[int] = ..., seed: _Optional[int] = ...) -> None: ...

class FineTuneSpec(_message.Message):
    __slots__ = ("source", "columns", "method", "task")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    source: str
    columns: _containers.RepeatedScalarFieldContainer[str]
    method: FineTuneMethod
    task: _inference_pb2.ModelTask
    def __init__(self, source: _Optional[str] = ..., columns: _Optional[_Iterable[str]] = ..., method: _Optional[_Union[FineTuneMethod, str]] = ..., task: _Optional[_Union[_inference_pb2.ModelTask, str]] = ...) -> None: ...

class GraphFineTuneSpec(_message.Message):
    __slots__ = ("sources", "sample_config")
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_CONFIG_FIELD_NUMBER: _ClassVar[int]
    sources: GraphFineTuneSources
    sample_config: GraphSampleConfig
    def __init__(self, sources: _Optional[_Union[GraphFineTuneSources, _Mapping]] = ..., sample_config: _Optional[_Union[GraphSampleConfig, _Mapping]] = ...) -> None: ...

class ContextPredictorSpec(_message.Message):
    __slots__ = ("source", "predictor_spec")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    PREDICTOR_SPEC_FIELD_NUMBER: _ClassVar[int]
    source: str
    predictor_spec: ContextPredictorTrainConfig
    def __init__(self, source: _Optional[str] = ..., predictor_spec: _Optional[_Union[ContextPredictorTrainConfig, _Mapping]] = ...) -> None: ...

class StartTrainingRequest(_message.Message):
    __slots__ = ("fine_tune", "graph_fine_tune", "context_predictor", "base_model", "config")
    FINE_TUNE_FIELD_NUMBER: _ClassVar[int]
    GRAPH_FINE_TUNE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_PREDICTOR_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    fine_tune: FineTuneSpec
    graph_fine_tune: GraphFineTuneSpec
    context_predictor: ContextPredictorSpec
    base_model: str
    config: FineTuneConfig
    def __init__(self, fine_tune: _Optional[_Union[FineTuneSpec, _Mapping]] = ..., graph_fine_tune: _Optional[_Union[GraphFineTuneSpec, _Mapping]] = ..., context_predictor: _Optional[_Union[ContextPredictorSpec, _Mapping]] = ..., base_model: _Optional[str] = ..., config: _Optional[_Union[FineTuneConfig, _Mapping]] = ...) -> None: ...

class StartTrainingResponse(_message.Message):
    __slots__ = ("job_id", "model_id")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    model_id: str
    def __init__(self, job_id: _Optional[str] = ..., model_id: _Optional[str] = ...) -> None: ...

class TrainingStatusRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class TrainingStatusResponse(_message.Message):
    __slots__ = ("status", "model_id", "error")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    status: str
    model_id: str
    error: str
    def __init__(self, status: _Optional[str] = ..., model_id: _Optional[str] = ..., error: _Optional[str] = ...) -> None: ...
