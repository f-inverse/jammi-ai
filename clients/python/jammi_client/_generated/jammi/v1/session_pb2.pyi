from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

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
