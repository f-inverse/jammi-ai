import datetime

from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

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
    def __init__(self, topic_id: _Optional[str] = ..., if_exists: _Optional[bool] = ...) -> None: ...

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
    __slots__ = ("topic", "predicate", "from_offset", "tenant_id")
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    FROM_OFFSET_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    topic: TopicName
    predicate: str
    from_offset: int
    tenant_id: str
    def __init__(self, topic: _Optional[_Union[TopicName, _Mapping]] = ..., predicate: _Optional[str] = ..., from_offset: _Optional[int] = ..., tenant_id: _Optional[str] = ...) -> None: ...

class SubscribedBatch(_message.Message):
    __slots__ = ("offset", "produced_at", "batch")
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    PRODUCED_AT_FIELD_NUMBER: _ClassVar[int]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    offset: int
    produced_at: _timestamp_pb2.Timestamp
    batch: ArrowBatch
    def __init__(self, offset: _Optional[int] = ..., produced_at: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., batch: _Optional[_Union[ArrowBatch, _Mapping]] = ...) -> None: ...

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
