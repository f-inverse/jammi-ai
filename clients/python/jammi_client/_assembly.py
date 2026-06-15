"""Kwargs → `jammi.v1` proto request assembly — the transport-agnostic builder layer.

This is the pure-Python request-construction shared above the wire: the enum maps
that translate a verb's snake-case vocabulary into the generated proto enums, the
small builder helpers that shape oneof/nested messages, and the per-verb request
builders that assemble a whole `StartTrainingRequest` from flat kwargs. It is
protobuf-only — no ML dependency, no transport — so both the pure gRPC client and
the embedded wheel's remote binding construct identical requests from one place.

Only request-side assembly lives here. Response decoding (wire → dict / table) and
the transport itself stay with the consumer that owns the channel.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ._generated.jammi.v1 import catalog_pb2
from ._generated.jammi.v1 import embedding_pb2
from ._generated.jammi.v1 import eval_pb2
from ._generated.jammi.v1 import inference_pb2
from ._generated.jammi.v1 import pipeline_pb2
from ._generated.jammi.v1 import training_pb2

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

# Inference-eval task string → the wire `EvalTask` enum, matching the engine's
# `EvalTask` parse (the two inference tasks that carry golden labels).
_EVAL_TASK = {
    "classification": eval_pb2.EvalTask.EVAL_TASK_CLASSIFICATION,
    "ner": eval_pb2.EvalTask.EVAL_TASK_NER,
}

# Channel-column dtype token → the wire `ChannelColumnType` enum. The token is
# the canonical PascalCase variant name the engine's `ChannelColumnType` parses
# (`ChannelColumnType::from_sql_str`) and the embed binding's `register_channel`
# accepts, so `register_channel(columns=[(name, dtype)])` takes the same
# vocabulary on both transports.
_CHANNEL_COLUMN_TYPE = {
    "Float32": catalog_pb2.ChannelColumnType.CHANNEL_COLUMN_TYPE_FLOAT32,
    "Float64": catalog_pb2.ChannelColumnType.CHANNEL_COLUMN_TYPE_FLOAT64,
    "Int32": catalog_pb2.ChannelColumnType.CHANNEL_COLUMN_TYPE_INT32,
    "Int64": catalog_pb2.ChannelColumnType.CHANNEL_COLUMN_TYPE_INT64,
    "Utf8": catalog_pb2.ChannelColumnType.CHANNEL_COLUMN_TYPE_UTF8,
    "Boolean": catalog_pb2.ChannelColumnType.CHANNEL_COLUMN_TYPE_BOOLEAN,
}


def _channel_columns_message(
    columns: Sequence[Tuple[str, str]],
) -> List[catalog_pb2.ChannelColumn]:
    """Encode `(name, dtype)` tuples into wire `ChannelColumn` messages.

    `dtype` is the canonical PascalCase token (`"Float32"`, `"Utf8"`, …) the
    embed binding's `register_channel` accepts; an unknown token raises
    `ValueError` rather than silently sending `UNSPECIFIED`, mirroring the
    engine's `ChannelColumnType::from_sql_str` rejection.
    """
    out: List[catalog_pb2.ChannelColumn] = []
    for name, dtype in columns:
        try:
            data_type = _CHANNEL_COLUMN_TYPE[dtype]
        except KeyError:
            raise ValueError(
                f"channel column dtype must be one of "
                f"{sorted(_CHANNEL_COLUMN_TYPE)} (got {dtype!r})"
            ) from None
        out.append(catalog_pb2.ChannelColumn(name=name, data_type=data_type))
    return out


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


def _regression_loss_message(
    regression_loss: Optional[str],
    *,
    regression_beta: Optional[float] = None,
) -> Optional[training_pb2.RegressionLoss]:
    """Build the wire `RegressionLoss` oneof from the embed binding's named loss.

    Mirrors the embed binding's `fine_tune` regression decoding exactly
    (`crates/jammi-python/src/database.rs`): the named loss with its scalar knob
    (`beta_nll` carries `regression_beta`, defaulting to 0.5), or `None` to let
    the engine select its collapse-resistant β-NLL default. An unnamed loss with
    only `regression_beta` set keeps the beta-implies-β-NLL shorthand. The
    proto's per-variant message names map onto the wire `RegressionLoss` oneof
    (`gaussian_nll`/`beta_nll`/`crps`/`pinball`, field 27); the wire validator
    enforces the beta range and quantile-level constraints before training.
    """
    if regression_loss is None:
        if regression_beta is not None:
            return training_pb2.RegressionLoss(
                beta_nll=training_pb2.RegressionLoss.BetaNll(beta=regression_beta)
            )
        return None
    if regression_loss == "gaussian_nll":
        return training_pb2.RegressionLoss(
            gaussian_nll=training_pb2.RegressionLoss.GaussianNll()
        )
    if regression_loss == "beta_nll":
        return training_pb2.RegressionLoss(
            beta_nll=training_pb2.RegressionLoss.BetaNll(
                beta=regression_beta if regression_beta is not None else 0.5
            )
        )
    if regression_loss == "crps":
        return training_pb2.RegressionLoss(crps=training_pb2.RegressionLoss.Crps())
    if regression_loss == "pinball":
        return training_pb2.RegressionLoss(
            pinball=training_pb2.RegressionLoss.Pinball()
        )
    raise ValueError(
        f"Unknown regression_loss {regression_loss!r}. Use 'gaussian_nll', "
        f"'beta_nll', 'crps', or 'pinball'."
    )


def build_fine_tune_config(
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
    seed: Optional[int],
    regression_loss: Optional[str],
    regression_beta: Optional[float],
    quantile_levels: Optional[List[float]],
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
    if seed is not None:
        config.seed = seed
    # Regression objective (task="regression" only), field 27. Mirrors the
    # embed binding's decoding: a named loss with its β knob, or — with only
    # `regression_beta` set — the beta-implies-β-NLL shorthand. Left UNSET
    # when neither is given so the server applies the engine's β-NLL default.
    regression = _regression_loss_message(
        regression_loss, regression_beta=regression_beta
    )
    if regression is not None:
        config.regression_loss.CopyFrom(regression)
    # Quantile levels (field 28) for a pinball-trained head; wired regardless
    # of the named loss so a Pinball request is reachable. Empty leaves the
    # field unset for the parametric Gaussian objectives.
    if quantile_levels is not None:
        config.quantile_levels.extend(quantile_levels)
    return config


def build_fine_tune_request(
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
) -> training_pb2.StartTrainingRequest:
    """Assemble the `StartTrainingRequest` for a LoRA fine-tune (the `FineTuneSpec`
    arm) from the embed binding's flat kwargs.

    Validates the `method` vocabulary, defaults `task` to text-embedding, builds
    the `FineTuneConfig` from the config kwargs, and wraps both in the request —
    the same shape the embed binding's `fine_tune` submits.
    """
    try:
        wire_method = _FINE_TUNE_METHOD[method]
    except KeyError:
        raise ValueError(
            f"method must be one of {sorted(_FINE_TUNE_METHOD)} (got {method!r})"
        ) from None
    wire_task = _MODEL_TASK[task] if task is not None else _MODEL_TASK["text_embedding"]

    config = build_fine_tune_config(
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
    return training_pb2.StartTrainingRequest(
        fine_tune=training_pb2.FineTuneSpec(
            source=source,
            columns=list(columns),
            method=wire_method,
            task=wire_task,
        ),
        base_model=base_model,
        config=config,
    )


def build_fine_tune_graph_request(
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
) -> training_pb2.StartTrainingRequest:
    """Assemble the `StartTrainingRequest` for a graph-supervised fine-tune (S11,
    the `GraphFineTuneSpec` arm) from the embed binding's flat kwargs.

    Validates the `edge_provenance` vocabulary, fills the `GraphSampleConfig`
    defaults (matching the engine's `GraphSampleConfig::default()`), and applies
    the graph-only embedding-loss guard (only `mnrl`/`triplet` for graph
    supervision). `edge_provenance` is the load-bearing circularity distinction —
    "declared" external edges teach the metric something new; "similarity" edges
    are a weak bootstrap only.
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
    # The graph sampler's `sample_seed` and the LoRA `seed` are distinct: one
    # seeds the node2vec walk, the other the adapter init / dropout.
    if seed is not None:
        config.seed = seed

    return training_pb2.StartTrainingRequest(
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
        config=config,
    )


def build_context_predictor_request(
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
) -> training_pb2.StartTrainingRequest:
    """Assemble the `StartTrainingRequest` for an amortized in-context predictor
    (S19, the `ContextPredictorSpec` arm) from the embed binding's flat kwargs.

    Validates the `architecture` vocabulary, builds the gaussian/quantile
    predictive head (with the `output='quantile' requires levels` check), and
    defaults the model id to `{source}-context-predictor`.
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
    return training_pb2.StartTrainingRequest(
        context_predictor=training_pb2.ContextPredictorSpec(
            source=source,
            predictor_spec=spec,
        )
    )
