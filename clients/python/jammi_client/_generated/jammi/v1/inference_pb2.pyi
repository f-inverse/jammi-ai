from . import trigger_pb2 as _trigger_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ModelTask(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODEL_TASK_UNSPECIFIED: _ClassVar[ModelTask]
    TEXT_EMBEDDING: _ClassVar[ModelTask]
    IMAGE_EMBEDDING: _ClassVar[ModelTask]
    AUDIO_EMBEDDING: _ClassVar[ModelTask]
    CLASSIFICATION: _ClassVar[ModelTask]
    NER: _ClassVar[ModelTask]
MODEL_TASK_UNSPECIFIED: ModelTask
TEXT_EMBEDDING: ModelTask
IMAGE_EMBEDDING: ModelTask
AUDIO_EMBEDDING: ModelTask
CLASSIFICATION: ModelTask
NER: ModelTask

class InferRequest(_message.Message):
    __slots__ = ("source_id", "model_id", "task", "columns", "key_column", "tenant_id")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    COLUMNS_FIELD_NUMBER: _ClassVar[int]
    KEY_COLUMN_FIELD_NUMBER: _ClassVar[int]
    TENANT_ID_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    model_id: str
    task: ModelTask
    columns: _containers.RepeatedScalarFieldContainer[str]
    key_column: str
    tenant_id: str
    def __init__(self, source_id: _Optional[str] = ..., model_id: _Optional[str] = ..., task: _Optional[_Union[ModelTask, str]] = ..., columns: _Optional[_Iterable[str]] = ..., key_column: _Optional[str] = ..., tenant_id: _Optional[str] = ...) -> None: ...

class InferResponse(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: _trigger_pb2.ArrowBatch
    def __init__(self, result: _Optional[_Union[_trigger_pb2.ArrowBatch, _Mapping]] = ...) -> None: ...
