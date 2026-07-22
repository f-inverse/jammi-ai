from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ChannelColumnType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHANNEL_COLUMN_TYPE_UNSPECIFIED: _ClassVar[ChannelColumnType]
    CHANNEL_COLUMN_TYPE_FLOAT32: _ClassVar[ChannelColumnType]
    CHANNEL_COLUMN_TYPE_FLOAT64: _ClassVar[ChannelColumnType]
    CHANNEL_COLUMN_TYPE_INT32: _ClassVar[ChannelColumnType]
    CHANNEL_COLUMN_TYPE_INT64: _ClassVar[ChannelColumnType]
    CHANNEL_COLUMN_TYPE_UTF8: _ClassVar[ChannelColumnType]
    CHANNEL_COLUMN_TYPE_BOOLEAN: _ClassVar[ChannelColumnType]
CHANNEL_COLUMN_TYPE_UNSPECIFIED: ChannelColumnType
CHANNEL_COLUMN_TYPE_FLOAT32: ChannelColumnType
CHANNEL_COLUMN_TYPE_FLOAT64: ChannelColumnType
CHANNEL_COLUMN_TYPE_INT32: ChannelColumnType
CHANNEL_COLUMN_TYPE_INT64: ChannelColumnType
CHANNEL_COLUMN_TYPE_UTF8: ChannelColumnType
CHANNEL_COLUMN_TYPE_BOOLEAN: ChannelColumnType

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
