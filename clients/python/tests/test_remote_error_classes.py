"""A remote failure raises the same `jammi.errors` class the embedded engine
raises: a typed engine detail on the status selects its leaf class, and a status
with none falls back to the class its gRPC code maps to."""

from __future__ import annotations

import grpc
from google.protobuf import any_pb2

from jammi._database import _rpc_to_jammi
from jammi._generated.jammi.v1 import error_pb2
from jammi.errors import (
    AlreadyExists,
    BackendError,
    FailedPrecondition,
    InvalidArgument,
    ModelNotFound,
    ModelReferenced,
    NotFound,
)


class _FailedCall(grpc.RpcError):
    """A failed call as the gRPC runtime reports it: a code, a message, and the
    trailing metadata the server sent."""

    def __init__(self, code, message, trailers=()):
        self._code, self._message, self._trailers = code, message, trailers

    def code(self):
        return self._code

    def details(self):
        return self._message

    def trailing_metadata(self):
        return self._trailers


def _with_detail(code, detail: error_pb2.JammiErrorDetail) -> _FailedCall:
    packed = any_pb2.Any()
    packed.Pack(detail)
    envelope = error_pb2.RpcStatus(code=code.value[0], message="m", details=[packed])
    return _FailedCall(code, "m", (("grpc-status-details-bin", envelope.SerializeToString()),))


def test_a_typed_detail_raises_its_leaf_class() -> None:
    referenced = error_pb2.JammiErrorDetail(model_referenced=error_pb2.ModelReferencedError())
    missing = error_pb2.JammiErrorDetail(model_not_found=error_pb2.ModelNotFoundError())

    raised = _rpc_to_jammi(_with_detail(grpc.StatusCode.FAILED_PRECONDITION, referenced))
    assert isinstance(raised, ModelReferenced) and isinstance(raised, FailedPrecondition)
    assert raised.code == grpc.StatusCode.FAILED_PRECONDITION
    assert isinstance(_rpc_to_jammi(_with_detail(grpc.StatusCode.NOT_FOUND, missing)), ModelNotFound)


def test_without_a_detail_the_code_decides() -> None:
    assert type(_rpc_to_jammi(_FailedCall(grpc.StatusCode.NOT_FOUND, "m"))) is NotFound
    assert type(_rpc_to_jammi(_FailedCall(grpc.StatusCode.ALREADY_EXISTS, "m"))) is AlreadyExists
    assert isinstance(
        _rpc_to_jammi(_FailedCall(grpc.StatusCode.FAILED_PRECONDITION, "m")), FailedPrecondition
    )
    assert type(_rpc_to_jammi(_FailedCall(grpc.StatusCode.INTERNAL, "m"))) is BackendError
    assert isinstance(
        _rpc_to_jammi(_FailedCall(grpc.StatusCode.INVALID_ARGUMENT, "m")), InvalidArgument
    )
