import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TopicName(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class ArrowBatch(_message.Message):
    __slots__ = ("data_header", "data_body", "app_metadata")
    DATA_HEADER_FIELD_NUMBER: _ClassVar[int]
    DATA_BODY_FIELD_NUMBER: _ClassVar[int]
    APP_METADATA_FIELD_NUMBER: _ClassVar[int]
    data_header: bytes
    data_body: bytes
    app_metadata: bytes
    def __init__(self, data_header: _Optional[bytes] = ..., data_body: _Optional[bytes] = ..., app_metadata: _Optional[bytes] = ...) -> None: ...

class PublishRequest(_message.Message):
    __slots__ = ("topic", "batch", "tenant_id")
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    topic: TopicName
    batch: ArrowBatch
    tenant_id: str
    def __init__(self, topic: _Optional[_Union[TopicName, _Mapping]] = ..., batch: _Optional[_Union[ArrowBatch, _Mapping]] = ..., tenant_id: _Optional[str] = ...) -> None: ...

class PublishResponse(_message.Message):
    __slots__ = ("offset", "committed_at")
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_AT_FIELD_NUMBER: _ClassVar[int]
    offset: int
    committed_at: _timestamp_pb2.Timestamp
    def __init__(self, offset: _Optional[int] = ..., committed_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class SubscribeRequest(_message.Message):
    __slots__ = ("topic", "predicate", "from_offset", "tenant_id", "replay_only")
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    FROM_OFFSET_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    REPLAY_ONLY_FIELD_NUMBER: _ClassVar[int]
    topic: TopicName
    predicate: str
    from_offset: int
    tenant_id: str
    replay_only: bool
    def __init__(self, topic: _Optional[_Union[TopicName, _Mapping]] = ..., predicate: _Optional[str] = ..., from_offset: _Optional[int] = ..., tenant_id: _Optional[str] = ..., replay_only: bool = ...) -> None: ...

class SubscribedBatch(_message.Message):
    __slots__ = ("offset", "produced_at", "batch")
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    PRODUCED_AT_FIELD_NUMBER: _ClassVar[int]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    offset: int
    produced_at: _timestamp_pb2.Timestamp
    batch: ArrowBatch
    def __init__(self, offset: _Optional[int] = ..., produced_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., batch: _Optional[_Union[ArrowBatch, _Mapping]] = ...) -> None: ...
