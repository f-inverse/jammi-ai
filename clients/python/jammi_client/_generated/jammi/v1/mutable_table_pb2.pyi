from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MutableIndex(_message.Message):
    __slots__ = ("name", "columns", "unique")
    NAME_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    UNIQUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    columns: _containers.RepeatedScalarFieldContainer[str]
    unique: bool
    def __init__(self, name: _Optional[str] = ..., columns: _Optional[_Iterable[str]] = ..., unique: _Optional[bool] = ...) -> None: ...

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
