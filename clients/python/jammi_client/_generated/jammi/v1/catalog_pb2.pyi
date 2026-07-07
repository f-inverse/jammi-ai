from google.protobuf import empty_pb2 as _empty_pb2
from . import inference_pb2 as _inference_pb2
from . import embedding_pb2 as _embedding_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SourceKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SOURCE_KIND_UNSPECIFIED: _ClassVar[SourceKind]
    SOURCE_KIND_FILE: _ClassVar[SourceKind]
    SOURCE_KIND_POSTGRES: _ClassVar[SourceKind]
    SOURCE_KIND_MYSQL: _ClassVar[SourceKind]

class FileFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FILE_FORMAT_UNSPECIFIED: _ClassVar[FileFormat]
    FILE_FORMAT_PARQUET: _ClassVar[FileFormat]
    FILE_FORMAT_CSV: _ClassVar[FileFormat]
    FILE_FORMAT_JSON: _ClassVar[FileFormat]
    FILE_FORMAT_AVRO: _ClassVar[FileFormat]

class ChannelColumnType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHANNEL_COLUMN_TYPE_UNSPECIFIED: _ClassVar[ChannelColumnType]
    CHANNEL_COLUMN_TYPE_FLOAT32: _ClassVar[ChannelColumnType]
    CHANNEL_COLUMN_TYPE_FLOAT64: _ClassVar[ChannelColumnType]
    CHANNEL_COLUMN_TYPE_INT32: _ClassVar[ChannelColumnType]
    CHANNEL_COLUMN_TYPE_INT64: _ClassVar[ChannelColumnType]
    CHANNEL_COLUMN_TYPE_UTF8: _ClassVar[ChannelColumnType]
    CHANNEL_COLUMN_TYPE_BOOLEAN: _ClassVar[ChannelColumnType]

class AnchorKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ANCHOR_KIND_UNSPECIFIED: _ClassVar[AnchorKind]
    ANCHOR_KIND_RESULT_DIGEST: _ClassVar[AnchorKind]
    ANCHOR_KIND_MUTABLE_VERSION: _ClassVar[AnchorKind]
    ANCHOR_KIND_SOURCE_VERSION: _ClassVar[AnchorKind]
    ANCHOR_KIND_UNPINNED_AT_INSTANT: _ClassVar[AnchorKind]
SOURCE_KIND_UNSPECIFIED: SourceKind
SOURCE_KIND_FILE: SourceKind
SOURCE_KIND_POSTGRES: SourceKind
SOURCE_KIND_MYSQL: SourceKind
FILE_FORMAT_UNSPECIFIED: FileFormat
FILE_FORMAT_PARQUET: FileFormat
FILE_FORMAT_CSV: FileFormat
FILE_FORMAT_JSON: FileFormat
FILE_FORMAT_AVRO: FileFormat
CHANNEL_COLUMN_TYPE_UNSPECIFIED: ChannelColumnType
CHANNEL_COLUMN_TYPE_FLOAT32: ChannelColumnType
CHANNEL_COLUMN_TYPE_FLOAT64: ChannelColumnType
CHANNEL_COLUMN_TYPE_INT32: ChannelColumnType
CHANNEL_COLUMN_TYPE_INT64: ChannelColumnType
CHANNEL_COLUMN_TYPE_UTF8: ChannelColumnType
CHANNEL_COLUMN_TYPE_BOOLEAN: ChannelColumnType
ANCHOR_KIND_UNSPECIFIED: AnchorKind
ANCHOR_KIND_RESULT_DIGEST: AnchorKind
ANCHOR_KIND_MUTABLE_VERSION: AnchorKind
ANCHOR_KIND_SOURCE_VERSION: AnchorKind
ANCHOR_KIND_UNPINNED_AT_INSTANT: AnchorKind

class ServerInfo(_message.Message):
    __slots__ = ("version", "features", "storage_backends", "services")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    STORAGE_BACKENDS_FIELD_NUMBER: _ClassVar[int]
    SERVICES_FIELD_NUMBER: _ClassVar[int]
    version: str
    features: _containers.RepeatedScalarFieldContainer[str]
    storage_backends: _containers.RepeatedScalarFieldContainer[str]
    services: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, version: _Optional[str] = ..., features: _Optional[_Iterable[str]] = ..., storage_backends: _Optional[_Iterable[str]] = ..., services: _Optional[_Iterable[str]] = ...) -> None: ...

class Tenant(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class SetTenantRequest(_message.Message):
    __slots__ = ("tenant",)
    TENANT_FIELD_NUMBER: _ClassVar[int]
    tenant: Tenant
    def __init__(self, tenant: _Optional[_Union[Tenant, _Mapping]] = ...) -> None: ...

class GetTenantResponse(_message.Message):
    __slots__ = ("tenant",)
    TENANT_FIELD_NUMBER: _ClassVar[int]
    tenant: Tenant
    def __init__(self, tenant: _Optional[_Union[Tenant, _Mapping]] = ...) -> None: ...

class SourceConnection(_message.Message):
    __slots__ = ("url", "format")
    URL_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    url: str
    format: FileFormat
    def __init__(self, url: _Optional[str] = ..., format: _Optional[_Union[FileFormat, str]] = ...) -> None: ...

class AddSourceRequest(_message.Message):
    __slots__ = ("source_id", "source_kind", "connection")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_KIND_FIELD_NUMBER: _ClassVar[int]
    CONNECTION_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    source_kind: SourceKind
    connection: SourceConnection
    def __init__(self, source_id: _Optional[str] = ..., source_kind: _Optional[_Union[SourceKind, str]] = ..., connection: _Optional[_Union[SourceConnection, _Mapping]] = ...) -> None: ...

class RemoveSourceRequest(_message.Message):
    __slots__ = ("source_id",)
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    def __init__(self, source_id: _Optional[str] = ...) -> None: ...

class ListSourcesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListSourcesResponse(_message.Message):
    __slots__ = ("sources",)
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    sources: _containers.RepeatedCompositeFieldContainer[SourceDescriptor]
    def __init__(self, sources: _Optional[_Iterable[_Union[SourceDescriptor, _Mapping]]] = ...) -> None: ...

class DescribeSourceRequest(_message.Message):
    __slots__ = ("source_id",)
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    def __init__(self, source_id: _Optional[str] = ...) -> None: ...

class SourceDescriptor(_message.Message):
    __slots__ = ("source_id", "kind", "status", "result_tables")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    RESULT_TABLES_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    kind: SourceKind
    status: str
    result_tables: _containers.RepeatedCompositeFieldContainer[_embedding_pb2.ResultTable]
    def __init__(self, source_id: _Optional[str] = ..., kind: _Optional[_Union[SourceKind, str]] = ..., status: _Optional[str] = ..., result_tables: _Optional[_Iterable[_Union[_embedding_pb2.ResultTable, _Mapping]]] = ...) -> None: ...

class ListModelsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListModelsResponse(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[Model]
    def __init__(self, models: _Optional[_Iterable[_Union[Model, _Mapping]]] = ...) -> None: ...

class DescribeModelRequest(_message.Message):
    __slots__ = ("model_id",)
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    def __init__(self, model_id: _Optional[str] = ...) -> None: ...

class DeleteModelRequest(_message.Message):
    __slots__ = ("model_id", "version", "if_exists")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    IF_EXISTS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    version: int
    if_exists: bool
    def __init__(self, model_id: _Optional[str] = ..., version: _Optional[int] = ..., if_exists: bool = ...) -> None: ...

class Model(_message.Message):
    __slots__ = ("model_id", "backend", "task", "status")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    BACKEND_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    backend: str
    task: _inference_pb2.ModelTask
    status: str
    def __init__(self, model_id: _Optional[str] = ..., backend: _Optional[str] = ..., task: _Optional[_Union[_inference_pb2.ModelTask, str]] = ..., status: _Optional[str] = ...) -> None: ...

class ChannelColumn(_message.Message):
    __slots__ = ("name", "data_type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    data_type: ChannelColumnType
    def __init__(self, name: _Optional[str] = ..., data_type: _Optional[_Union[ChannelColumnType, str]] = ...) -> None: ...

class RegisterChannelRequest(_message.Message):
    __slots__ = ("channel_id", "priority", "columns")
    CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    channel_id: str
    priority: int
    columns: _containers.RepeatedCompositeFieldContainer[ChannelColumn]
    def __init__(self, channel_id: _Optional[str] = ..., priority: _Optional[int] = ..., columns: _Optional[_Iterable[_Union[ChannelColumn, _Mapping]]] = ...) -> None: ...

class AddChannelColumnsRequest(_message.Message):
    __slots__ = ("channel_id", "columns")
    CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    channel_id: str
    columns: _containers.RepeatedCompositeFieldContainer[ChannelColumn]
    def __init__(self, channel_id: _Optional[str] = ..., columns: _Optional[_Iterable[_Union[ChannelColumn, _Mapping]]] = ...) -> None: ...

class VerifyMaterializationRequest(_message.Message):
    __slots__ = ("table", "expected_definition")
    TABLE_FIELD_NUMBER: _ClassVar[int]
    EXPECTED_DEFINITION_FIELD_NUMBER: _ClassVar[int]
    table: str
    expected_definition: str
    def __init__(self, table: _Optional[str] = ..., expected_definition: _Optional[str] = ...) -> None: ...

class VerifyMaterializationResponse(_message.Message):
    __slots__ = ("match", "mismatch", "match_with_unpinned_inputs", "missing_manifest")
    class Match(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class Mismatch(_message.Message):
        __slots__ = ("expected", "found")
        EXPECTED_FIELD_NUMBER: _ClassVar[int]
        FOUND_FIELD_NUMBER: _ClassVar[int]
        expected: str
        found: str
        def __init__(self, expected: _Optional[str] = ..., found: _Optional[str] = ...) -> None: ...
    class MatchWithUnpinnedInputs(_message.Message):
        __slots__ = ("unpinned",)
        UNPINNED_FIELD_NUMBER: _ClassVar[int]
        unpinned: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, unpinned: _Optional[_Iterable[str]] = ...) -> None: ...
    class MissingManifest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    MATCH_FIELD_NUMBER: _ClassVar[int]
    MISMATCH_FIELD_NUMBER: _ClassVar[int]
    MATCH_WITH_UNPINNED_INPUTS_FIELD_NUMBER: _ClassVar[int]
    MISSING_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    match: VerifyMaterializationResponse.Match
    mismatch: VerifyMaterializationResponse.Mismatch
    match_with_unpinned_inputs: VerifyMaterializationResponse.MatchWithUnpinnedInputs
    missing_manifest: VerifyMaterializationResponse.MissingManifest
    def __init__(self, match: _Optional[_Union[VerifyMaterializationResponse.Match, _Mapping]] = ..., mismatch: _Optional[_Union[VerifyMaterializationResponse.Mismatch, _Mapping]] = ..., match_with_unpinned_inputs: _Optional[_Union[VerifyMaterializationResponse.MatchWithUnpinnedInputs, _Mapping]] = ..., missing_manifest: _Optional[_Union[VerifyMaterializationResponse.MissingManifest, _Mapping]] = ...) -> None: ...

class StalenessRequest(_message.Message):
    __slots__ = ("table", "current_definition")
    TABLE_FIELD_NUMBER: _ClassVar[int]
    CURRENT_DEFINITION_FIELD_NUMBER: _ClassVar[int]
    table: str
    current_definition: str
    def __init__(self, table: _Optional[str] = ..., current_definition: _Optional[str] = ...) -> None: ...

class StaleReason(_message.Message):
    __slots__ = ("definition_changed", "input_advanced", "input_vanished")
    class DefinitionChanged(_message.Message):
        __slots__ = ("recorded", "current")
        RECORDED_FIELD_NUMBER: _ClassVar[int]
        CURRENT_FIELD_NUMBER: _ClassVar[int]
        recorded: str
        current: str
        def __init__(self, recorded: _Optional[str] = ..., current: _Optional[str] = ...) -> None: ...
    class InputAdvanced(_message.Message):
        __slots__ = ("source", "recorded", "current")
        SOURCE_FIELD_NUMBER: _ClassVar[int]
        RECORDED_FIELD_NUMBER: _ClassVar[int]
        CURRENT_FIELD_NUMBER: _ClassVar[int]
        source: str
        recorded: str
        current: str
        def __init__(self, source: _Optional[str] = ..., recorded: _Optional[str] = ..., current: _Optional[str] = ...) -> None: ...
    class InputVanished(_message.Message):
        __slots__ = ("source",)
        SOURCE_FIELD_NUMBER: _ClassVar[int]
        source: str
        def __init__(self, source: _Optional[str] = ...) -> None: ...
    DEFINITION_CHANGED_FIELD_NUMBER: _ClassVar[int]
    INPUT_ADVANCED_FIELD_NUMBER: _ClassVar[int]
    INPUT_VANISHED_FIELD_NUMBER: _ClassVar[int]
    definition_changed: StaleReason.DefinitionChanged
    input_advanced: StaleReason.InputAdvanced
    input_vanished: StaleReason.InputVanished
    def __init__(self, definition_changed: _Optional[_Union[StaleReason.DefinitionChanged, _Mapping]] = ..., input_advanced: _Optional[_Union[StaleReason.InputAdvanced, _Mapping]] = ..., input_vanished: _Optional[_Union[StaleReason.InputVanished, _Mapping]] = ...) -> None: ...

class StalenessResponse(_message.Message):
    __slots__ = ("fresh", "stale", "undecidable", "missing_manifest")
    class Fresh(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    class Stale(_message.Message):
        __slots__ = ("reasons",)
        REASONS_FIELD_NUMBER: _ClassVar[int]
        reasons: _containers.RepeatedCompositeFieldContainer[StaleReason]
        def __init__(self, reasons: _Optional[_Iterable[_Union[StaleReason, _Mapping]]] = ...) -> None: ...
    class Undecidable(_message.Message):
        __slots__ = ("unpinned", "decided_reasons")
        UNPINNED_FIELD_NUMBER: _ClassVar[int]
        DECIDED_REASONS_FIELD_NUMBER: _ClassVar[int]
        unpinned: _containers.RepeatedScalarFieldContainer[str]
        decided_reasons: _containers.RepeatedCompositeFieldContainer[StaleReason]
        def __init__(self, unpinned: _Optional[_Iterable[str]] = ..., decided_reasons: _Optional[_Iterable[_Union[StaleReason, _Mapping]]] = ...) -> None: ...
    class MissingManifest(_message.Message):
        __slots__ = ()
        def __init__(self) -> None: ...
    FRESH_FIELD_NUMBER: _ClassVar[int]
    STALE_FIELD_NUMBER: _ClassVar[int]
    UNDECIDABLE_FIELD_NUMBER: _ClassVar[int]
    MISSING_MANIFEST_FIELD_NUMBER: _ClassVar[int]
    fresh: StalenessResponse.Fresh
    stale: StalenessResponse.Stale
    undecidable: StalenessResponse.Undecidable
    missing_manifest: StalenessResponse.MissingManifest
    def __init__(self, fresh: _Optional[_Union[StalenessResponse.Fresh, _Mapping]] = ..., stale: _Optional[_Union[StalenessResponse.Stale, _Mapping]] = ..., undecidable: _Optional[_Union[StalenessResponse.Undecidable, _Mapping]] = ..., missing_manifest: _Optional[_Union[StalenessResponse.MissingManifest, _Mapping]] = ...) -> None: ...

class DerivesFromRequest(_message.Message):
    __slots__ = ("table",)
    TABLE_FIELD_NUMBER: _ClassVar[int]
    table: str
    def __init__(self, table: _Optional[str] = ...) -> None: ...

class DerivesFromEdge(_message.Message):
    __slots__ = ("input", "derived", "kind")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    DERIVED_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    input: str
    derived: str
    kind: AnchorKind
    def __init__(self, input: _Optional[str] = ..., derived: _Optional[str] = ..., kind: _Optional[_Union[AnchorKind, str]] = ...) -> None: ...

class DerivesFromResponse(_message.Message):
    __slots__ = ("edges",)
    EDGES_FIELD_NUMBER: _ClassVar[int]
    edges: _containers.RepeatedCompositeFieldContainer[DerivesFromEdge]
    def __init__(self, edges: _Optional[_Iterable[_Union[DerivesFromEdge, _Mapping]]] = ...) -> None: ...

class ListChannelsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListChannelsResponse(_message.Message):
    __slots__ = ("channels",)
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    channels: _containers.RepeatedCompositeFieldContainer[Channel]
    def __init__(self, channels: _Optional[_Iterable[_Union[Channel, _Mapping]]] = ...) -> None: ...

class Channel(_message.Message):
    __slots__ = ("channel_id", "priority", "columns")
    CHANNEL_ID_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    channel_id: str
    priority: int
    columns: _containers.RepeatedCompositeFieldContainer[ChannelColumn]
    def __init__(self, channel_id: _Optional[str] = ..., priority: _Optional[int] = ..., columns: _Optional[_Iterable[_Union[ChannelColumn, _Mapping]]] = ...) -> None: ...

class MutableIndex(_message.Message):
    __slots__ = ("name", "columns", "unique")
    NAME_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    UNIQUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    columns: _containers.RepeatedScalarFieldContainer[str]
    unique: bool
    def __init__(self, name: _Optional[str] = ..., columns: _Optional[_Iterable[str]] = ..., unique: bool = ...) -> None: ...

class MutableTableDefinition(_message.Message):
    __slots__ = ("id", "schema", "primary_key", "indexes", "order_column", "chunk_size", "user_metadata")
    ID_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    PRIMARY_KEY_FIELD_NUMBER: _ClassVar[int]
    INDEXES_FIELD_NUMBER: _ClassVar[int]
    ORDER_COLUMN_FIELD_NUMBER: _ClassVar[int]
    CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    USER_METADATA_FIELD_NUMBER: _ClassVar[int]
    id: str
    schema: bytes
    primary_key: _containers.RepeatedScalarFieldContainer[str]
    indexes: _containers.RepeatedCompositeFieldContainer[MutableIndex]
    order_column: str
    chunk_size: int
    user_metadata: str
    def __init__(self, id: _Optional[str] = ..., schema: _Optional[bytes] = ..., primary_key: _Optional[_Iterable[str]] = ..., indexes: _Optional[_Iterable[_Union[MutableIndex, _Mapping]]] = ..., order_column: _Optional[str] = ..., chunk_size: _Optional[int] = ..., user_metadata: _Optional[str] = ...) -> None: ...

class CreateMutableTableRequest(_message.Message):
    __slots__ = ("definition",)
    DEFINITION_FIELD_NUMBER: _ClassVar[int]
    definition: MutableTableDefinition
    def __init__(self, definition: _Optional[_Union[MutableTableDefinition, _Mapping]] = ...) -> None: ...

class CreateMutableTableResponse(_message.Message):
    __slots__ = ("mutable_table_id",)
    MUTABLE_TABLE_ID_FIELD_NUMBER: _ClassVar[int]
    mutable_table_id: str
    def __init__(self, mutable_table_id: _Optional[str] = ...) -> None: ...

class DropMutableTableRequest(_message.Message):
    __slots__ = ("mutable_table_id",)
    MUTABLE_TABLE_ID_FIELD_NUMBER: _ClassVar[int]
    mutable_table_id: str
    def __init__(self, mutable_table_id: _Optional[str] = ...) -> None: ...

class ListMutableTablesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListMutableTablesResponse(_message.Message):
    __slots__ = ("definitions",)
    DEFINITIONS_FIELD_NUMBER: _ClassVar[int]
    definitions: _containers.RepeatedCompositeFieldContainer[MutableTableDefinition]
    def __init__(self, definitions: _Optional[_Iterable[_Union[MutableTableDefinition, _Mapping]]] = ...) -> None: ...

class RegisterTopicRequest(_message.Message):
    __slots__ = ("name", "schema", "broker_metadata", "topic_id")
    class BrokerMetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    BROKER_METADATA_FIELD_NUMBER: _ClassVar[int]
    TOPIC_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    schema: bytes
    broker_metadata: _containers.ScalarMap[str, str]
    topic_id: str
    def __init__(self, name: _Optional[str] = ..., schema: _Optional[bytes] = ..., broker_metadata: _Optional[_Mapping[str, str]] = ..., topic_id: _Optional[str] = ...) -> None: ...

class RegisterTopicResponse(_message.Message):
    __slots__ = ("topic_id",)
    TOPIC_ID_FIELD_NUMBER: _ClassVar[int]
    topic_id: str
    def __init__(self, topic_id: _Optional[str] = ...) -> None: ...

class DropTopicRequest(_message.Message):
    __slots__ = ("topic_id", "if_exists")
    TOPIC_ID_FIELD_NUMBER: _ClassVar[int]
    IF_EXISTS_FIELD_NUMBER: _ClassVar[int]
    topic_id: str
    if_exists: bool
    def __init__(self, topic_id: _Optional[str] = ..., if_exists: bool = ...) -> None: ...

class ListTopicsRequest(_message.Message):
    __slots__ = ("page_size", "page_token", "tenant_id")
    PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    page_size: int
    page_token: str
    tenant_id: str
    def __init__(self, page_size: _Optional[int] = ..., page_token: _Optional[str] = ..., tenant_id: _Optional[str] = ...) -> None: ...

class Topic(_message.Message):
    __slots__ = ("topic_id", "name", "schema", "tenant_id", "broker_metadata")
    class BrokerMetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TOPIC_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    BROKER_METADATA_FIELD_NUMBER: _ClassVar[int]
    topic_id: str
    name: str
    schema: bytes
    tenant_id: str
    broker_metadata: _containers.ScalarMap[str, str]
    def __init__(self, topic_id: _Optional[str] = ..., name: _Optional[str] = ..., schema: _Optional[bytes] = ..., tenant_id: _Optional[str] = ..., broker_metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class ListTopicsResponse(_message.Message):
    __slots__ = ("topics", "next_page_token")
    TOPICS_FIELD_NUMBER: _ClassVar[int]
    NEXT_PAGE_TOKEN_FIELD_NUMBER: _ClassVar[int]
    topics: _containers.RepeatedCompositeFieldContainer[Topic]
    next_page_token: str
    def __init__(self, topics: _Optional[_Iterable[_Union[Topic, _Mapping]]] = ..., next_page_token: _Optional[str] = ...) -> None: ...
