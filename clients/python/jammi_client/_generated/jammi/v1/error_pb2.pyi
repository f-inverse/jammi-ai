from google.protobuf import any_pb2 as _any_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RpcStatus(_message.Message):
    __slots__ = ("code", "message", "details")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    code: int
    message: str
    details: _containers.RepeatedCompositeFieldContainer[_any_pb2.Any]
    def __init__(self, code: _Optional[int] = ..., message: _Optional[str] = ..., details: _Optional[_Iterable[_Union[_any_pb2.Any, _Mapping]]] = ...) -> None: ...

class JammiErrorDetail(_message.Message):
    __slots__ = ("source", "model", "inference", "catalog", "schema", "config", "eval", "tenant", "other", "fine_tune", "gpu", "backend", "channel_catalog", "mutable_table", "channel_assembly", "model_referenced", "model_not_found")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_FIELD_NUMBER: _ClassVar[int]
    CATALOG_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    EVAL_FIELD_NUMBER: _ClassVar[int]
    TENANT_FIELD_NUMBER: _ClassVar[int]
    OTHER_FIELD_NUMBER: _ClassVar[int]
    FINE_TUNE_FIELD_NUMBER: _ClassVar[int]
    GPU_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_CATALOG_FIELD_NUMBER: _ClassVar[int]
    MUTABLE_TABLE_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_ASSEMBLY_FIELD_NUMBER: _ClassVar[int]
    MODEL_REFERENCED_FIELD_NUMBER: _ClassVar[int]
    MODEL_NOT_FOUND_FIELD_NUMBER: _ClassVar[int]
    source: SourceError
    model: ModelError
    inference: StringError
    catalog: StringError
    schema: SchemaError
    config: StringError
    eval: StringError
    tenant: StringError
    other: StringError
    fine_tune: StringError
    gpu: StringError
    backend: StringError
    channel_catalog: ChannelCatalogErrorDetail
    mutable_table: MutableTableErrorDetail
    channel_assembly: StringError
    model_referenced: ModelReferencedError
    model_not_found: ModelNotFoundError
    def __init__(self, source: _Optional[_Union[SourceError, _Mapping]] = ..., model: _Optional[_Union[ModelError, _Mapping]] = ..., inference: _Optional[_Union[StringError, _Mapping]] = ..., catalog: _Optional[_Union[StringError, _Mapping]] = ..., schema: _Optional[_Union[SchemaError, _Mapping]] = ..., config: _Optional[_Union[StringError, _Mapping]] = ..., eval: _Optional[_Union[StringError, _Mapping]] = ..., tenant: _Optional[_Union[StringError, _Mapping]] = ..., other: _Optional[_Union[StringError, _Mapping]] = ..., fine_tune: _Optional[_Union[StringError, _Mapping]] = ..., gpu: _Optional[_Union[StringError, _Mapping]] = ..., backend: _Optional[_Union[StringError, _Mapping]] = ..., channel_catalog: _Optional[_Union[ChannelCatalogErrorDetail, _Mapping]] = ..., mutable_table: _Optional[_Union[MutableTableErrorDetail, _Mapping]] = ..., channel_assembly: _Optional[_Union[StringError, _Mapping]] = ..., model_referenced: _Optional[_Union[ModelReferencedError, _Mapping]] = ..., model_not_found: _Optional[_Union[ModelNotFoundError, _Mapping]] = ...) -> None: ...

class ChannelCatalogErrorDetail(_message.Message):
    __slots__ = ("already_exists", "not_registered", "column_already_declared", "column_conflict", "invalid_id", "invalid_column_type")
    ALREADY_EXISTS_FIELD_NUMBER: _ClassVar[int]
    NOT_REGISTERED_FIELD_NUMBER: _ClassVar[int]
    COLUMN_ALREADY_DECLARED_FIELD_NUMBER: _ClassVar[int]
    COLUMN_CONFLICT_FIELD_NUMBER: _ClassVar[int]
    INVALID_ID_FIELD_NUMBER: _ClassVar[int]
    INVALID_COLUMN_TYPE_FIELD_NUMBER: _ClassVar[int]
    already_exists: str
    not_registered: str
    column_already_declared: ColumnAlreadyDeclared
    column_conflict: ColumnConflict
    invalid_id: str
    invalid_column_type: str
    def __init__(self, already_exists: _Optional[str] = ..., not_registered: _Optional[str] = ..., column_already_declared: _Optional[_Union[ColumnAlreadyDeclared, _Mapping]] = ..., column_conflict: _Optional[_Union[ColumnConflict, _Mapping]] = ..., invalid_id: _Optional[str] = ..., invalid_column_type: _Optional[str] = ...) -> None: ...

class ColumnAlreadyDeclared(_message.Message):
    __slots__ = ("channel", "column", "ty")
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    COLUMN_FIELD_NUMBER: _ClassVar[int]
    TY_FIELD_NUMBER: _ClassVar[int]
    channel: str
    column: str
    ty: str
    def __init__(self, channel: _Optional[str] = ..., column: _Optional[str] = ..., ty: _Optional[str] = ...) -> None: ...

class ColumnConflict(_message.Message):
    __slots__ = ("channel", "column", "existing", "requested")
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    COLUMN_FIELD_NUMBER: _ClassVar[int]
    EXISTING_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_FIELD_NUMBER: _ClassVar[int]
    channel: str
    column: str
    existing: str
    requested: str
    def __init__(self, channel: _Optional[str] = ..., column: _Optional[str] = ..., existing: _Optional[str] = ..., requested: _Optional[str] = ...) -> None: ...

class ModelReferencedError(_message.Message):
    __slots__ = ("model_id", "referenced_by")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    REFERENCED_BY_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    referenced_by: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, model_id: _Optional[str] = ..., referenced_by: _Optional[_Iterable[str]] = ...) -> None: ...

class ModelNotFoundError(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class MutableTableErrorDetail(_message.Message):
    __slots__ = ("invalid_id", "schema", "missing_primary_key", "reserved_column", "not_found", "already_exists", "no_order_column", "backend")
    INVALID_ID_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    MISSING_PRIMARY_KEY_FIELD_NUMBER: _ClassVar[int]
    RESERVED_COLUMN_FIELD_NUMBER: _ClassVar[int]
    NOT_FOUND_FIELD_NUMBER: _ClassVar[int]
    ALREADY_EXISTS_FIELD_NUMBER: _ClassVar[int]
    NO_ORDER_COLUMN_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    invalid_id: str
    schema: str
    missing_primary_key: str
    reserved_column: str
    not_found: str
    already_exists: str
    no_order_column: bool
    backend: BackendErrorDetail
    def __init__(self, invalid_id: _Optional[str] = ..., schema: _Optional[str] = ..., missing_primary_key: _Optional[str] = ..., reserved_column: _Optional[str] = ..., not_found: _Optional[str] = ..., already_exists: _Optional[str] = ..., no_order_column: bool = ..., backend: _Optional[_Union[BackendErrorDetail, _Mapping]] = ...) -> None: ...

class BackendErrorDetail(_message.Message):
    __slots__ = ("execution", "constraint", "unavailable", "retry", "migration", "type_conversion", "tenant_mismatch", "sqlx")
    EXECUTION_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINT_FIELD_NUMBER: _ClassVar[int]
    UNAVAILABLE_FIELD_NUMBER: _ClassVar[int]
    RETRY_FIELD_NUMBER: _ClassVar[int]
    MIGRATION_FIELD_NUMBER: _ClassVar[int]
    TYPE_CONVERSION_FIELD_NUMBER: _ClassVar[int]
    TENANT_MISMATCH_FIELD_NUMBER: _ClassVar[int]
    SQLX_FIELD_NUMBER: _ClassVar[int]
    execution: str
    constraint: ConstraintViolation
    unavailable: str
    retry: str
    migration: str
    type_conversion: TypeConversion
    tenant_mismatch: TenantMismatch
    sqlx: str
    def __init__(self, execution: _Optional[str] = ..., constraint: _Optional[_Union[ConstraintViolation, _Mapping]] = ..., unavailable: _Optional[str] = ..., retry: _Optional[str] = ..., migration: _Optional[str] = ..., type_conversion: _Optional[_Union[TypeConversion, _Mapping]] = ..., tenant_mismatch: _Optional[_Union[TenantMismatch, _Mapping]] = ..., sqlx: _Optional[str] = ...) -> None: ...

class ConstraintViolation(_message.Message):
    __slots__ = ("table", "detail")
    TABLE_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    table: str
    detail: str
    def __init__(self, table: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...

class TypeConversion(_message.Message):
    __slots__ = ("column", "detail")
    COLUMN_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    column: str
    detail: str
    def __init__(self, column: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...

class TenantMismatch(_message.Message):
    __slots__ = ("table", "expected", "got")
    TABLE_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_FIELD_NUMBER: _ClassVar[int]
    GOT_FIELD_NUMBER: _ClassVar[int]
    table: str
    expected: str
    got: str
    def __init__(self, table: _Optional[str] = ..., expected: _Optional[str] = ..., got: _Optional[str] = ...) -> None: ...

class StringError(_message.Message):
    __slots__ = ("message",)
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class SourceError(_message.Message):
    __slots__ = ("source_id", "message")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    message: str
    def __init__(self, source_id: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class ModelError(_message.Message):
    __slots__ = ("model_id", "message")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    message: str
    def __init__(self, model_id: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class SchemaError(_message.Message):
    __slots__ = ("table", "column", "expected", "actual")
    TABLE_FIELD_NUMBER: _ClassVar[int]
    COLUMN_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_FIELD_NUMBER: _ClassVar[int]
    table: str
    column: str
    expected: str
    actual: str
    def __init__(self, table: _Optional[str] = ..., column: _Optional[str] = ..., expected: _Optional[str] = ..., actual: _Optional[str] = ...) -> None: ...

class TriggerErrorDetail(_message.Message):
    __slots__ = ("topic_not_found", "schema_conflict", "unsupported_schema_type", "batch_schema_mismatch", "publish_tenant_mismatch", "predicate_parse", "predicate_eval", "predicate_unsupported", "offset_evicted", "backing_table", "backend", "driver", "catalog")
    TOPIC_NOT_FOUND_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_CONFLICT_FIELD_NUMBER: _ClassVar[int]
    UNSUPPORTED_SCHEMA_TYPE_FIELD_NUMBER: _ClassVar[int]
    BATCH_SCHEMA_MISMATCH_FIELD_NUMBER: _ClassVar[int]
    PUBLISH_TENANT_MISMATCH_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_PARSE_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_EVAL_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_UNSUPPORTED_FIELD_NUMBER: _ClassVar[int]
    OFFSET_EVICTED_FIELD_NUMBER: _ClassVar[int]
    BACKING_TABLE_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    DRIVER_FIELD_NUMBER: _ClassVar[int]
    CATALOG_FIELD_NUMBER: _ClassVar[int]
    topic_not_found: str
    schema_conflict: SchemaConflict
    unsupported_schema_type: UnsupportedSchemaType
    batch_schema_mismatch: str
    publish_tenant_mismatch: PublishTenantMismatch
    predicate_parse: str
    predicate_eval: str
    predicate_unsupported: str
    offset_evicted: int
    backing_table: MutableTableErrorDetail
    backend: BackendErrorDetail
    driver: str
    catalog: str
    def __init__(self, topic_not_found: _Optional[str] = ..., schema_conflict: _Optional[_Union[SchemaConflict, _Mapping]] = ..., unsupported_schema_type: _Optional[_Union[UnsupportedSchemaType, _Mapping]] = ..., batch_schema_mismatch: _Optional[str] = ..., publish_tenant_mismatch: _Optional[_Union[PublishTenantMismatch, _Mapping]] = ..., predicate_parse: _Optional[str] = ..., predicate_eval: _Optional[str] = ..., predicate_unsupported: _Optional[str] = ..., offset_evicted: _Optional[int] = ..., backing_table: _Optional[_Union[MutableTableErrorDetail, _Mapping]] = ..., backend: _Optional[_Union[BackendErrorDetail, _Mapping]] = ..., driver: _Optional[str] = ..., catalog: _Optional[str] = ...) -> None: ...

class SchemaConflict(_message.Message):
    __slots__ = ("topic", "detail")
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    topic: str
    detail: str
    def __init__(self, topic: _Optional[str] = ..., detail: _Optional[str] = ...) -> None: ...

class UnsupportedSchemaType(_message.Message):
    __slots__ = ("column", "data_type")
    COLUMN_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    column: str
    data_type: str
    def __init__(self, column: _Optional[str] = ..., data_type: _Optional[str] = ...) -> None: ...

class PublishTenantMismatch(_message.Message):
    __slots__ = ("topic", "topic_tenant", "publish_tenant")
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    TOPIC_TENANT_FIELD_NUMBER: _ClassVar[int]
    PUBLISH_TENANT_FIELD_NUMBER: _ClassVar[int]
    topic: str
    topic_tenant: str
    publish_tenant: str
    def __init__(self, topic: _Optional[str] = ..., topic_tenant: _Optional[str] = ..., publish_tenant: _Optional[str] = ...) -> None: ...

class AuditErrorDetail(_message.Message):
    __slots__ = ("length_mismatch", "lineage_too_large", "no_tenant_binding", "signature_mismatch", "master_key", "serde", "storage", "broker")
    LENGTH_MISMATCH_FIELD_NUMBER: _ClassVar[int]
    LINEAGE_TOO_LARGE_FIELD_NUMBER: _ClassVar[int]
    NO_TENANT_BINDING_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_MISMATCH_FIELD_NUMBER: _ClassVar[int]
    MASTER_KEY_FIELD_NUMBER: _ClassVar[int]
    SERDE_FIELD_NUMBER: _ClassVar[int]
    STORAGE_FIELD_NUMBER: _ClassVar[int]
    BROKER_FIELD_NUMBER: _ClassVar[int]
    length_mismatch: LengthMismatch
    lineage_too_large: LineageTooLarge
    no_tenant_binding: bool
    signature_mismatch: str
    master_key: str
    serde: str
    storage: str
    broker: str
    def __init__(self, length_mismatch: _Optional[_Union[LengthMismatch, _Mapping]] = ..., lineage_too_large: _Optional[_Union[LineageTooLarge, _Mapping]] = ..., no_tenant_binding: bool = ..., signature_mismatch: _Optional[str] = ..., master_key: _Optional[str] = ..., serde: _Optional[str] = ..., storage: _Optional[str] = ..., broker: _Optional[str] = ...) -> None: ...

class LengthMismatch(_message.Message):
    __slots__ = ("ids", "scores")
    IDS_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    ids: int
    scores: int
    def __init__(self, ids: _Optional[int] = ..., scores: _Optional[int] = ...) -> None: ...

class LineageTooLarge(_message.Message):
    __slots__ = ("actual", "max")
    ACTUAL_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    actual: int
    max: int
    def __init__(self, actual: _Optional[int] = ..., max: _Optional[int] = ...) -> None: ...
